from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import csv


def run(train_set: Path, test_set: Path, delta_t: int, sigma: float, out_path: Path):
    train_mean = []
    test_mean = []
    train_var = []
    test_var = []
    clrs = sns.color_palette("Set2", 2)
    with open(train_set, mode="r") as fd_r:
        reader = csv.DictReader(fd_r)
        for row in reader:
            train_mean.append(float(row["data_mean"]))
            train_var.append(float(row["data_var"]))
    with open(test_set, mode="r") as fd_r:
        reader = csv.DictReader(fd_r)
        for row in reader:
            test_mean.append(float(row["data_mean"]))
            test_var.append(float(row["data_var"]))

    x_train = np.arange(delta_t, (len(train_mean) + 1) * delta_t, delta_t)
    x_test = np.arange(delta_t, (len(test_mean) + 1) * delta_t, delta_t)

    train_var = np.array(train_var, dtype=np.float32)
    train_mean = np.array(train_mean, dtype=np.float32)
    train_std = np.sqrt(train_var)

    test_var = np.array(test_var, dtype=np.float32)
    test_mean = np.array(test_mean, dtype=np.float32)
    test_std = np.sqrt(test_var)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots()
        ax.plot(x_train, train_mean, label=f"Train return", c=clrs[0])
        ax.fill_between(x_train, train_mean - sigma * train_std,
                        train_mean + sigma * train_std, interpolate=True, alpha=0.3, facecolor=clrs[0])
        ax.plot(x_test, test_mean, label=f"Test return", c=clrs[1])
        ax.fill_between(x_test, test_mean - sigma * test_std,
                        test_mean + sigma * test_std, interpolate=True, alpha=0.3, facecolor=clrs[1])
        legend = ax.legend(frameon=False)
        legend.get_frame().set_facecolor('none')
        plt.title(f"Pair association task return by number of training episodes \n(noise = N(0,0.1) \n (μ ±{sigma}σ)")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
