import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from multiclass_evaluator import MetricsCalculator
from incre_space_data_stream import incre_stream
from subspace_classifier import SubspaceResidualTrapezoidalClassifier
from matplotlib import colors



dataset_name = "agrawal_gradual"
data_path = f"data/{dataset_name}.csv"

subspace_grid = [2, 3, 4, 5, 6, 7, 8, 9]

js_threshold_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

n_repeats = 1

init_n = 500

chunk_size = 500


result_dir = "result"
os.makedirs(result_dir, exist_ok=True)
csv_path = os.path.join(result_dir, f"sensitivity_{dataset_name}.csv")



def run_one_setting(file_path: str,
                    n_subspaces: int,
                    js_threshold: float,
                    repeat_seed: int = 0) -> float:

    stream = incre_stream(file_path, change_times=5)


    inc_data, label = stream.next_sample(init_n)

    classes = stream.target_values

    clf = SubspaceResidualTrapezoidalClassifier(
        n_subspaces=n_subspaces,
        init_size=init_n,
        window_size_L=chunk_size,
        js_threshold=js_threshold
    )


    clf.partial_fit(inc_data, label, classes=classes)

    evaluator = MetricsCalculator(classes=classes)


    while stream.has_more_samples():
        inc_data, label = stream.next_sample()

        y_pred = clf.predict(inc_data)
        evaluator.add_result(prediction=y_pred, label=label)


        clf.partial_fit(inc_data, label, classes=classes)

    avg_err = evaluator.calculate_accuracy()
    return avg_err


def run_sensitivity_and_save():
    rows = []
    for k in subspace_grid:
        for js_th in js_threshold_grid:
            errs = []
            for r in range(n_repeats):
                err = run_one_setting(
                    file_path=data_path,
                    n_subspaces=k,
                    js_threshold=js_th,
                    repeat_seed=r
                )
                errs.append(err)

            rows.append({
                "dataset": dataset_name,
                "n_subspaces": k,
                "js_threshold": js_th,
                "avg_error": float(np.mean(errs)),
                "std_error": float(np.std(errs))
            })
            print(f"[OK] K={k:<2d}  JS={js_th:<5.3f}  avg_error={np.mean(errs):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")



def plot_from_csv(csv_file: str):
    df = pd.read_csv(csv_file)

    ks = sorted(df["n_subspaces"].unique())
    js = sorted(df["js_threshold"].unique())


    js_idx = list(range(len(js)))

    K_grid, JSIDX_grid = np.meshgrid(ks, js_idx)
    Z = np.zeros_like(K_grid, dtype=float)

    for i, js_th in enumerate(js):
        for j, k in enumerate(ks):
            z = df[(df["n_subspaces"] == k) &
                   (df["js_threshold"] == js_th)]["avg_error"]
            Z[i, j] = float(z.values[0]) if len(z) else np.nan


    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    norm = colors.Normalize(vmin=z_min, vmax=z_max)


    fig = plt.figure(figsize=(8.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        K_grid,
        JSIDX_grid,
        Z,
        cmap="viridis",
        norm=norm,
        linewidth=0.35,
        antialiased=True
    )


    ax.set_xlabel(r"$K$", labelpad=8)
    ax.set_ylabel(r"$\delta$", labelpad=10)


    ax.set_yticks(js_idx)
    ax.set_yticklabels([f"{v:.1e}" for v in js])
    ax.tick_params(axis="y", pad=4)
    ax.tick_params(axis="z", pad=4)


    ax.view_init(elev=28, azim=-60)


    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.02, top=0.98)


    cbar = fig.colorbar(surf, shrink=0.6, aspect=18)
    cbar.set_label("Classification accuracy")


    plt.savefig(
        f"fig/sensitivity_{dataset_name}.jpg",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.close()



if __name__ == "__main__":

    if not os.path.exists(csv_path):
        run_sensitivity_and_save()
    else:
        print(f"CSV already exists: {csv_path} (skip running)")

    plot_from_csv(csv_path)
