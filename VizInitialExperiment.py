from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

def graph_feature(results, feature, widths=None, when="last", moving_average=False):
    if widths == None:
        widths = sorted(list(results.keys()), key=int)   # allows you to only plot some widths
    if when == "last":
        y = np.array([results[i][feature][-1] for i in widths])
    elif when == "best":
        y = np.array([min(results[i][feature]) for i in widths])
    else:
        raise Exception("Argument when must be 'last' or 'best'")
    if moving_average:
        def calculate_ema(arr):
            arr_return = [arr[0]]
            for x in arr[1:]:
                arr_return.append(arr_return[-1]*moving_average + x*(1-moving_average))
            return arr_return
        y = calculate_ema(y)
    fig, ax = plt.subplots()
    ax.scatter(widths, y)
    ax.set_xlabel("Width")
    ax.set_ylabel(feature + " after Completed Training")
    return fig

def graph_curve(results, feature, width, apply_log=True, moving_average=False):
    # feature MUST be "loss_curve", or some other similarly formatted array
    vals = torch.tensor(results[width][feature])
    if moving_average:
        def calculate_ema(arr):
            arr_return = [arr[0]]
            for x in arr[1:]:
                arr_return.append(arr_return[-1]*moving_average + x*(1-moving_average))
            return torch.tensor(arr_return)
        vals = calculate_ema(vals)
    fig, ax = plt.subplots()
    ax.plot(results[width]["epochs"], vals.detach().numpy())
    if apply_log:
        ax.set_yscale("log")
    ax.set_ylabel(feature)
    ax.set_title(feature + " curve for m = " + str(width) + " over training")
    ax.set_xlabel("Epoch")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="The path to the json results file saved by InitialExperiment.py")
    args = parser.parse_args()

    with open(args.results_file, 'r') as f:
        results = json.load(f)

    for feature in ["sharpness", "train_loss", "valid_loss"]:
        figure_tosave = graph_feature(results, feature, when="last", moving_average=0.9)
        (Path(args.results_file).parent / Path(args.results_file).stem).mkdir(parents=True, exist_ok=True)
        figure_tosave.savefig(Path(args.results_file).parent /
                               Path(args.results_file).stem /
                                (feature + "final" + ".png"),
                                bbox_inches="tight")
        
    for width in results.keys():
        for feature in ["sharpness", "train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]:
            if feature in ["sharpness", "train_loss", "valid_loss"]:
                figure_tosave = graph_curve(results, feature, width, moving_average=0.9)
            else:
                figure_tosave = graph_curve(results, feature, width, apply_log=False, moving_average=0.9)
            figure_tosave.savefig(Path(args.results_file).parent /
                                  Path(args.results_file).stem /
                                    (feature + "_" + str(width) + ".png"),
                                    bbox_inches="tight")
