import numpy as np
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from multiclass_evaluator import MetricsCalculator
import pandas as pd
import os
from subspace_classifier import SubspaceResidualTrapezoidalClassifier
from compare_mathods import IIFClassifier, RAILClassifier, IWEMclassifier
from incre_space_data_stream import incre_stream

file_name = ['agrawal_sudden', 'agrawal_gradual', 'stag_sudden', 'stag_gradual', 'tree_sudden', 'tree_gradual', 'rbf', 'wave', 'weather', 'elec', 'sensor', 'covtype']

accuracy = []
accuracy_div = []
for name in file_name:
    file_results_acc = []
    for _ in range(10):
        SALF = SubspaceResidualTrapezoidalClassifier(n_subspaces = 7, js_threshold = 0.001)
        IIF = IIFClassifier()
        RAIL = RAILClassifier()
        DWM = DynamicWeightedMajorityClassifier()
        IWEM = IWEMclassifier()
        DWM_A = DynamicWeightedMajorityClassifier()
        IWEM_A = IWEMclassifier()
        Models = [SALF, IIF, RAIL, DWM, IWEM, DWM_A, IWEM_A]

        file = "data/" + name + ".csv"
        stream = incre_stream(file, change_times=5)
        inc_data, fix_data, all_data, label = stream.next_sample(500, return_fixed=True, return_all=True)
        classes = stream.target_values
        for i in range(len(Models)):
            if i == 3 or i == 4:
                Models[i].partial_fit(fix_data, label, classes=classes)
            elif i == 5 or i == 6:
                Models[i].partial_fit(all_data, label, classes=classes)
            else:
                Models[i].partial_fit(inc_data, label, classes=classes)

        evaluators = []
        for i in range(len(Models)):
            matrix = MetricsCalculator(classes = classes)
            evaluators.append(matrix)

        while stream.has_more_samples():
            inc_data, fix_data, all_data, label = stream.next_sample(return_fixed=True, return_all=True)
            for i in range(len(Models)):
                if i == 3 or i == 4:
                    predict = Models[i].predict(fix_data)
                elif i == 5 or i == 6:
                    predict = Models[i].predict(all_data)
                else:
                    predict = Models[i].predict(inc_data)
                evaluators[i].add_result(prediction=predict, label=label)

                if i == 3 or i == 4:
                    Models[i].partial_fit(fix_data, label, classes=classes)
                elif i == 5 or i == 6:
                    Models[i].partial_fit(all_data, label, classes=classes)
                else:
                    Models[i].partial_fit(inc_data, label, classes=classes)

        line_acc = []
        for i in range(len(Models)):
            line_acc.append(evaluators[i].calculate_accuracy())
        file_results_acc.append(line_acc)
    accuracy.append(list(np.mean(np.array(file_results_acc), axis=0)))
    accuracy_div.append(list(np.std(np.array(file_results_acc), axis=0)))
    print(name)

accuracy = pd.DataFrame(accuracy)
accuracy_div = pd.DataFrame(accuracy_div)
accuracy.columns = ['SALF', 'IIF', 'RAIL', 'DWM', 'IWEM', 'DWM_A', 'IWEM_A']
accuracy_div.columns = ['SALF', 'IIF', 'RAIL', 'DWM', 'IWEM', 'DWM_A', 'IWEM_A']
accuracy.index = file_name
accuracy_div.index = file_name
folder_path = "result/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
accuracy.to_csv(folder_path + 'accuracy.csv')
accuracy_div.to_csv(folder_path + 'accuracy.div.csv')