from compare_mathods import IIFClassifier, RAILClassifier, IWEMclassifier
import pandas as pd
import os
from multiclass_evaluator import MetricsCalculator
from subspace_classifier import SubspaceResidualTrapezoidalClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from incre_space_data_stream import incre_stream

file_name = ['agrawal_sudden', 'agrawal_gradual', 'stag_sudden', 'stag_gradual', 'tree_sudden', 'tree_gradual', 'rbf', 'wave']

for name in file_name:
    SALF = SubspaceResidualTrapezoidalClassifier(n_subspaces = 7, js_threshold = 0.001)
    IIF = IIFClassifier()
    RAIL = RAILClassifier()
    DWM = DynamicWeightedMajorityClassifier()
    IWEM = IWEMclassifier()
    Models = [SALF, IIF, RAIL, DWM, IWEM]

    file = "data/" + name + ".csv"
    stream = incre_stream(file, change_times=5)
    inc_data, fix_data, label = stream.next_sample(500, return_fixed=True)
    classes = stream.target_values

    for i in range(len(Models)):
        if i == 3 or i == 4:
            Models[i].partial_fit(fix_data, label, classes=classes)
        else:
            Models[i].partial_fit(inc_data, label, classes=classes)

    evaluators = []
    time_accuracy = []
    for i in range(len(Models)):
        time_accuracy.append([])
        matrix = MetricsCalculator(classes=classes)
        evaluators.append(matrix)
    data_size = 500
    time_accuracy.append([])

    while stream.has_more_samples():
        inc_data, fix_data, label = stream.next_sample(return_fixed=True)
        data_size = data_size + 1
        for i in range(len(Models)):
            if i == 3 or i == 4:
                predict = Models[i].predict(fix_data)
            else:
                predict = Models[i].predict(inc_data)
            evaluators[i].add_result(prediction=predict, label=label)

            if i == 3 or i == 4:
                Models[i].partial_fit(fix_data, label, classes=classes)
            else:
                Models[i].partial_fit(inc_data, label, classes=classes)

        if data_size % 5000 == 0:
            for i in range(len(Models)):
                time_accuracy[i].append(evaluators[i].calculate_accuracy())
            evaluators = []
            for i in range(len(Models)):
                matrix = MetricsCalculator(classes=classes)
                evaluators.append(matrix)
            time_accuracy[-1].append(data_size)

    time_accuracy = pd.DataFrame(time_accuracy)
    folder_path = "result/curve/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    time_accuracy.to_csv(folder_path + name + '.csv')
    print('finish' + name)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_name = [
    'agrawal_sudden', 'agrawal_gradual',
    'stag_sudden', 'stag_gradual',
    'tree_sudden', 'tree_gradual',
    'rbf', 'wave'
]

def clean_cell(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace('[', '').replace(']', '').strip()
    if s == '':
        return np.nan
    return float(s)

def infer_x_from_columns(df):
    keys = ("time", "step", "timestep", "iter")
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in keys):
            return df[c].to_numpy(), c
    return None, None

label = ['SALF', 'IIF', 'RAIL', 'DWM', 'IWEM']
style = [':', '-.', '--', '-', '--']
point = ['.', ',', 'o', 'v', '*']


for stream in file_name:
    path = 'result/curve/' + stream + '.csv'
    df_raw = pd.read_csv(path)

    df = df_raw.apply(lambda s: s.map(clean_cell))

    x, x_col = infer_x_from_columns(df)

    data = df.values.tolist()

    first_col_name = str(df_raw.columns[0]).lower()
    drop_first_col = ("unnamed" in first_col_name) or (first_col_name in ("index", "idx"))

    if drop_first_col:
        for i in range(len(data)):
            if len(data[i]) > 0:
                data[i].pop(0)


    if x is None:
        if len(data) >= 7:
            x = np.array(data[6], dtype=float)
        else:

            if len(data) == 0:
                print(f"[WARN] {stream}: empty csv.")
                continue
            T = len(data[0])
            x = np.arange(T, dtype=float)


    plt.figure(figsize=(9, 3.6))

    for i, name in enumerate(label):

        if name in df_raw.columns:
            y = df[name].to_numpy(dtype=float)
        else:

            if i < len(data):
                y = np.array(data[i], dtype=float)
            else:
                print(f"[WARN] {stream}: cannot find curve for {name}. Skip this method.")
                continue

        L = min(len(x), len(y))
        xx = x[:L]
        yy = y[:L]

        if name == 'SALF':
            plt.plot(xx, yy, label=name, linewidth=1.8, color='r',
                     linestyle=style[i], marker=point[i])
        else:
            plt.plot(xx, yy, label=name, linewidth=1.6,
                     linestyle=style[i], marker=point[i])

    plt.legend(fontsize=12, loc='lower left')
    plt.xlabel('Time steps', size=15)
    plt.ylabel('Classification accuracy', size=15)

    plt.tight_layout()

    os.makedirs("fig", exist_ok=True)
    out_name = "fig/" + stream + "_curve.jpg"
    plt.savefig(out_name, dpi=350)
    plt.close()

    print(f"[OK] Saved: {out_name}")
