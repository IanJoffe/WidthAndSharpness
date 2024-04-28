from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

def graph_feature(results, feature, widths=None):
    if widths == None:
        widths = sorted(list(results.keys()))   # allows you to only plot some widths
    y = np.array([results[i][feature] for i in widths])
    fig, ax = plt.subplots()
    ax.scatter(widths, y)
    return fig

def graph_loss(results, feature, width):
    # feature MUST be "valid_loss", "train_loss", or some other similarly formatted array
    losses = torch.tensor(results[width][feature])
    fig, ax = plt.subplots()
    plt.plot(range(len(losses.detach().numpy())), np.log(losses.detach().numpy()))
    ax.scatter(widths, y)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="The path to the pickled results saved by InitialExperiment.py")
    args = parser.parse_args()

    with open(args.results_file, 'rb') as f:
        results = pickle.load(f)

    for feature in ["sharpness, train_loss", "valid_loss"]:
        figure_tosave = graph_feature(feature, results)
        figure_tosave.savefig(Path(args.results_file).parent /
                               Path(args.results_file).parent /
                                (feature + ".png"))
        
    for width in results.keys():
        figure_tosave = graph_loss(results, "train_loss", width)
        figure_tosave.savefig(Path(args.results_file).parent /
                               Path(args.results_file).parent /
                                ("train_loss_" + str(width) + ".png"))
        figure_tosave = graph_loss(results, "valid_loss", width)
        figure_tosave.savefig(Path(args.results_file).parent /
                               Path(args.results_file).parent /
                                ("valid_loss_" + str(width) + ".png"))
