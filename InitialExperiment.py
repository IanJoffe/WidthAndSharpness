import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sam import SAM
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import copy


class simpleDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]
  

class two_layer_relu_network(nn.Module):
    def __init__(self, input_size, output_size, width):
        super().__init__()
        self.fc1 = nn.Linear(input_size, width, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(width, output_size, bias=False)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        return z2
  

def train_true_model(d):
    """
    Train a model that may be used as the ground truth function in the experiment
    """
    def sin_sum(x):
        return torch.sin(torch.sum(x, dim=1).unsqueeze(1))

    torch.manual_seed(2024)
    input_data = torch.normal(mean=0, std=1, size=(100, d)).to(device)
    input_data = input_data / torch.norm(input_data, dim=0)
    output_data = sin_sum(input_data).to(device)
    dataset = simpleDataset(input_data, output_data)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    ground_truth_model = two_layer_relu_network(d, 1, 5000).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(ground_truth_model.parameters(), lr=3e-4)

    print("Training ground truth model")
    for epoch in np.arange(5000):       # NN empirically converges by this time
        all_loss = torch.tensor([]).to(device)
        for data, labels in dataloader:

            optimizer.zero_grad()
            model_output = ground_truth_model(data)
            label_noise = torch.normal(mean=0, std=0.05, size=(1,)).to(device)
            loss = criterion(model_output, labels + label_noise)
            loss.backward(retain_graph=True)
            optimizer.step()

            all_loss = torch.cat((all_loss, loss.unsqueeze(0)))

        if epoch % int(500) == 0:
            print("Loss: ", torch.mean(all_loss))
    print("Completed training ground truth model")

    return ground_truth_model
  
  
def run_width_experiment(n=100, n_valid=1000, d=10, m=list(range(10, 100, 5)),
                         lr = 3e-4, momentum=0, batch_size=1, use_sam=False,
                         convergence_req=1e-3, convergence_halt=False, max_epochs=3e4, num_measurements=200,
                         input_dist=torch.normal, input_dist_args = {"mean":0, "std":1}, normalize_input=True, shuffle_data=False,
                         true_function=lambda x: torch.sin(torch.sum(x, dim=1).unsqueeze(1)),
                         label_noise_sd=0.05, random_seed=137):
    """
    RETURNS: {m: model}, {m: [converged, sharpness, train_loss, valid_loss, mean_training_sparsity, mean_valid_sparsity, loss_curve]}
    ARGS:
    n: int, number of points in training data
    n_valid: int, number of points in validation data
    d: int, dimension of each training data point
    m: list[int], widths of neural network to run experiment on
    lr: float, learning rate
    momentum: float, momentum parameter for SGD
    use_sam: bool, set to true to use the same optimizer, if false will use SGD with label noise
    convergence_req: float, the NN will be considered converged if EVERY data points has loss this low, also used to calculate accuracy
    convergence_halt: bool, set to True to halt training once converged
    max_epochs: int, the NN will run for this many epochs before giving up on convergence
    num_measurements: int, validation loss and sharpness will be measured very max_epochs/num_measurements epochs
    input_dist: torch function, determines distribution of input data
    input_dist_args: dict{str: float}, a dictionary with parameters like mean and sd for the input_dist. Do not include the size parameter.
    normalize_input: bool, whether to normalize input trian and valid data to have norm 1. Useful to convert gaussian data to uniform on the hypersphere.
    shuffle_data: whether to shuffle the order of the (data, label) pairs before each epoch of SGD
    true_function: torch function, output = torch_function(input). The function should act on the full nxd matrix of input data
    label_noise_sd: float, gives the standard deviation for gaussian noise in noisy SGD
    random_seed: int
    """

    experiment_results = {}
    model_checkpoints = {}

    # generate data
    torch.manual_seed(random_seed)
    input_data = input_dist(**input_dist_args, size=(n, d)).to(device)
    if normalize_input:
        input_data = input_data / (torch.norm(input_data, dim=1).unsqueeze(1))
    output_data = true_function(input_data).to(device)

    valid_input = input_dist(**input_dist_args, size=(n_valid, d)).to(device)
    if normalize_input:
        valid_input = valid_input / (torch.norm(valid_input, dim=1).unsqueeze(1))
    valid_output = true_function(valid_input).to(device)

    dataset = simpleDataset(input_data, output_data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle_data)

    print("Training experimental models")
    # run NN for each width
    for width in m:
        model = two_layer_relu_network(d, 1, width).to(device)
        criterion = nn.MSELoss()
        if use_sam:
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=momentum)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss = np.inf
        converged = False
        epochs = max_epochs

        model_checkpoints[int(width)] = {}
        measured_epochs = []
        train_loss = []
        valid_loss = []
        train_accuracy = []
        valid_accuracy = []
        sharpness = []
        def get_sharpness(data, model):
            def point_sharpness(x, model):
                gradients = torch.autograd.grad(model(x), model.parameters(), create_graph=True)
                return torch.linalg.vector_norm(torch.cat([g.flatten() for g in gradients]))**2
            total_sharpness = 2/len(data) * np.sum(np.array([point_sharpness(x, model).item() for x in data]))
            return total_sharpness

        with tqdm(range(int(epochs)+1), desc="Training Progress, m=" + str(width)) as progress_bar:
            for epoch in progress_bar:

                # take measurments
                if epoch % int(epochs/num_measurements) == 0:
                    measured_epochs.append(epoch)
                    train_loss.append(criterion(model(input_data), output_data).item())
                    valid_loss.append(criterion(model(valid_input), valid_output).item())
                    train_accuracy.append(torch.isclose(model(input_data), output_data, atol=convergence_req).float().mean().item())
                    valid_accuracy.append(torch.isclose(model(valid_input), valid_output, atol=convergence_req).float().mean().item())
                    sharpness.append(get_sharpness(input_data, model))    # causes warning and won't run on cuda only for first epoch
                    model_checkpoints[int(width)][epoch] = copy.deepcopy(model)
                    
                all_loss = torch.tensor([]).to(device)
                for data, labels in dataloader:

                    #optimizer.zero_grad()
                    model_output = model(data)
                    label_noise = torch.normal(0, label_noise_sd, size=(batch_size,)).unsqueeze(dim=1).to(device)
                    if use_sam:
                        loss = criterion(model_output, labels)
                    else:
                        loss = criterion(model_output, labels + label_noise)

                    if epoch != epochs-1:
                        if use_sam:
                            loss.backward()
                            optimizer.first_step(zero_grad=True)
                            criterion(model(data), labels).backward()
                            optimizer.second_step(zero_grad=True)
                        else:
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                    loss_unnoisy = criterion(model_output, labels)
                    all_loss = torch.cat((all_loss, loss_unnoisy.unsqueeze(0)))

                if epoch % int(epochs/num_measurements) == 0:
                    progress_bar.set_postfix(avg_loss=torch.mean(all_loss).item(), max_loss=torch.max(all_loss).item())

                # check convergence
                if (not converged) and all(all_loss < convergence_req):
                    model_output = model(data)
                    loss = criterion(model_output, labels)
                    converged = True
                    print("Model with width", width, "interpolated in", epoch, "epochs")
                    if convergence_halt:
                        break
                if (not converged) and (epoch == epochs - 1):
                    print("Model with width", width, "did not interpolate")

        experiment_results[int(width)] = {"converged":converged, "epochs":measured_epochs, "train_loss":train_loss, "valid_loss":valid_loss, "train_accuracy":train_accuracy, "valid_accuracy":valid_accuracy, "sharpness": sharpness}

    print("Completed training experimental models")
    return model_checkpoints, experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="The path to the json file to save results to")
    args = parser.parse_args()
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.init()
    print("Connected to", str(device))

    # ground_truth_model = train_true_model(d=30)
    # XOR Function| lambda x: (x[:, 1] * x[:, 2]).unsqueeze(1)
    # Binary Data Generator| lambda size: torch.randint(0, 2, size=size, dtype=torch.float32)*2-1
    trained_models, results =  run_width_experiment(n=300, d=30, m=np.array([70]), use_sam=True, input_dist=lambda size: torch.randint(0, 2, size=size, dtype=torch.float32)*2-1, input_dist_args={}, normalize_input=False, shuffle_data=True, true_function=lambda x: (x[:, 1] * x[:, 2]).unsqueeze(1), convergence_req=1e-2, lr=1.2e-1, batch_size=150, label_noise_sd=0.25, max_epochs=1.2e6)
    for m in trained_models.keys():
        (Path(args.results_file).parent / ("checkpoints_" + str(m))).mkdir(parents=True, exist_ok=True)
        for epoch in trained_models[m].keys():
            torch.save(trained_models[m][epoch], Path(args.results_file).parent / ("checkpoints_" + str(m)) / ("model_" + str(m) + "_epoch_" + str(epoch) + ".pth"))
    with open(args.results_file, 'w') as f:
        json.dump(results, f)