from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

def graph_feature(results, feature, widths=None, when="last"):
    if widths == None:
        widths = sorted(list(results.keys()), key=int)   # allows you to only plot some widths
    if when == "last":
        y = np.array([results[i][feature][-1] for i in widths])
    elif when == "best":
        y = np.array([min(results[i][feature]) for i in widths])
    else:
        y = np.array([])
    fig, ax = plt.subplots()
    ax.scatter(widths, y)
    ax.set_xlabel("Width")
    ax.set_ylabel(feature + " after Completed Training")
    return fig

def graph_curve(results, feature, width, apply_log=True):
    # feature MUST be "loss_curve", or some other similarly formatted array
    vals = torch.tensor(results[width][feature])
    fig, ax = plt.subplots()
    if apply_log:
        ax.set_ylabel("Log " + feature)
        ax.plot(results[width]["epochs"], np.log(vals.detach().numpy()))
    else:
        ax.plot(results[width]["epochs"], vals.detach().numpy())
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
        figure_tosave = graph_feature(results, feature, when="last")
        (Path(args.results_file).parent / Path(args.results_file).stem).mkdir(parents=True, exist_ok=True)
        figure_tosave.savefig(Path(args.results_file).parent /
                               Path(args.results_file).stem /
                                (feature + "final" + ".png"),
                                bbox_inches="tight")
        
    for feature in ["sharpness", "train_loss", "valid_loss"]:
        figure_tosave = graph_feature(results, feature, when="best")
        (Path(args.results_file).parent / Path(args.results_file).stem).mkdir(parents=True, exist_ok=True)
        figure_tosave.savefig(Path(args.results_file).parent /
                               Path(args.results_file).stem /
                                (feature + "best" + ".png"),
                                bbox_inches="tight")
        
    for width in results.keys():
        for feature in ["sharpness", "train_loss", "valid_loss"]:
            figure_tosave = graph_curve(results, feature, width)
            figure_tosave.savefig(Path(args.results_file).parent /
                                  Path(args.results_file).stem /
                                    (feature + "_" + str(width) + ".png"),
                                    bbox_inches="tight")
