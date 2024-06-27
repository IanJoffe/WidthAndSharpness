from InitialExperiment import two_layer_relu_network
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

def graph_feature(results, feature, widths=None, when="last", moving_average=False, caption=None):
    if widths == None:
        widths = sorted(list(results.keys()), key=int)   # allows you to only plot some widths
    if when == "last":
        y = np.array([results[i][feature][-1] for i in widths])
    elif when == "best":
        y = np.array([min(results[i][feature][int(len(results[i][feature])/2):]) for i in widths])    # best may only occur in second half of measurements
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
    if caption:
        fig.text(0.5, -0.03, caption, ha='center', va='center')
    return fig

def graph_curve(results, feature, width, apply_log=True, moving_average=False, caption=None):
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
    if caption:
        fig.text(0.5, -0.03, caption, ha='center', va='center')
    return fig

def plot_weights_progression(checkpoints_list, results, width, min_dim=None, max_dim=None, min_neuron=None, max_neuron=None, caption=None):
    fc1_weights_list = [model.fc1.weight.data.detach().cpu().numpy() for model in checkpoints_list]
    fc1_weights_list_to_plot = np.array([weights[min_neuron:max_neuron,min_dim:max_dim].flatten() for weights in fc1_weights_list]).T
    fig, ax = plt.subplots()
    for weights in fc1_weights_list_to_plot:
        ax.plot(results[width]["epochs"], weights, alpha=0.1, c="blue")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Epoch")
    ax.set_title("Progression of FC1 Weights Over Training, m=" + str(width))
    if caption:
        fig.text(0.5, -0.03, caption, ha='center', va='center')
    return fig

def plot_final_weights(saved_model, width, min_dim=1, max_dim=3, caption=None):
    assert max_dim - min_dim == 2, "Cannot plot weights for d > 2"
    weights_fc1 = saved_model.fc1.weight.data.detach().cpu().numpy()
    informative_weights_fc1 = weights_fc1[:,min_dim:max_dim]
    fig, ax = plt.subplots()
    ax.scatter(informative_weights_fc1[:,0], informative_weights_fc1[:,1], s=5)
    ax.set_xlim(1.2 * min(min(informative_weights_fc1[:,0]), min(informative_weights_fc1[:,1])), 1.2 * max(max(informative_weights_fc1[:,0]), max(informative_weights_fc1[:,1])))
    ax.set_ylim(1.2 * min(min(informative_weights_fc1[:,0]), min(informative_weights_fc1[:,1])), 1.2 * max(max(informative_weights_fc1[:,0]), max(informative_weights_fc1[:,1])))
    ax.set_title("First Layer Weights for Informative Dimensions, m = " + str(width))
    if caption:
        fig.text(0.5, -0.03, caption, ha='center', va='center')
    return fig

def hist_weights(saved_model, width, min_dim=3, max_dim=None, caption=None):
    weights_fc1 = saved_model.fc1.weight.data.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.hist(weights_fc1[:,min_dim:max_dim].flatten(), bins=100)
    ax.set_title("First Layer Weights for Spurious Dimensions, m = " + str(width))
    if caption:
        fig.text(0.5, -0.03, caption, ha='center', va='center')
    return fig



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="The path to the json results file saved by InitialExperiment.py")
    parser.add_argument("--only", type=str, help="Only generate the named visualization. Can be one of or a comma-seperated list of 'feature', 'curve', 'progression', or 'weights'")
    args = parser.parse_args()

    if args.only:
        only = args.only.split(",")
    else:
        only=None
    all_results_files = args.results_file.split(",")
    results_file = all_results_files[0]
    if len(all_results_files) == 1:
        with open(all_results_files[0], 'r') as f:
            results = json.load(f)
    else:
        results = {}
        for results_file in all_results_files:
            with open(results_file, 'r') as f:
                current_results_file = json.load(f)
                if results.keys() & current_results_file.keys():
                    assert "Tried to merge results from two experiments with the same widths"
                results = results | current_results_file

    with open(Path(results_file).parent / "model_params.json", 'r') as f:
        caption = str(json.load(f))

    if not only or "feature" in only:
        for feature in ["sharpness", "train_loss", "valid_loss"]:
            figure_tosave = graph_feature(results, feature, when="best", moving_average=False, caption=caption)
            Path(results_file).parent.mkdir(parents=True, exist_ok=True)
            figure_tosave.savefig(Path(results_file).parent /
                                    (feature + "final" + ".png"),
                                    bbox_inches="tight")
    
    if not only or "curve" in only:
        for width in results.keys():
            for feature in ["sharpness", "train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]:
                if feature in ["sharpness", "train_loss", "valid_loss"]:
                    figure_tosave = graph_curve(results, feature, width, moving_average=False)
                else:
                    figure_tosave = graph_curve(results, feature, width, apply_log=False, moving_average=False)
                figure_tosave.savefig(Path(results_file).parent /
                                        (feature + "_" + str(width) + ".png"),
                                        bbox_inches="tight")
            
    if not only or "progression" in only:
        checkpoint_subdirs = [d for d in Path(results_file).parent.iterdir() if (d.is_dir() and d.name.startswith("checkpoints"))]
        for checkpoint_subdir in checkpoint_subdirs:
            width = checkpoint_subdir.name.split("_")[-1]
            checkpoints_list_fnames = sorted([ckpt for ckpt in checkpoint_subdir.iterdir()], key=lambda fname: int(fname.stem.split("_")[-1]))
            checkpoints_list = [torch.load(ckpt) for ckpt in checkpoints_list_fnames]
            figure_tosave = plot_weights_progression(checkpoints_list, results, width)
            figure_tosave.savefig(Path(results_file).parent /
                                ("weights_progression_" + str(width) + ".png"),
                                bbox_inches="tight")

    if not only or "weights" in only:
        for width in results.keys():
            all_checkpoints = [ckpt for ckpt in (Path(results_file).parent / ("checkpoints_" + str(width))).iterdir()]
            final_checkpoint = max(all_checkpoints, key=lambda fname: int(fname.stem.split("_")[-1]))
            saved_model = torch.load(final_checkpoint)
            figure_tosave = hist_weights(saved_model, width)
            figure_tosave.savefig(Path(results_file).parent /
                                    ("spurious_weights_" + str(width) + ".png"),
                                    bbox_inches="tight")
            figure_tosave = plot_final_weights(saved_model, width)
            figure_tosave.savefig(Path(results_file).parent /
                                    ("informative_weights_" + str(width) + ".png"),
                                    bbox_inches="tight")
            
    print("Visualizations saved to " + str(Path(results_file).parent))