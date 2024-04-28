import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


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
  

def train_true_model():
    """
    Train a model that may be used as the ground truth function in the experiment
    """
    def sin_sum(x):
        return torch.sin(torch.sum(x, dim=1).unsqueeze(1))

    torch.manual_seed(2024)
    input_data = torch.normal(mean=0, std=1, size=(100, 10)).to(device)
    input_data = input_data / torch.norm(input_data, dim=0)
    output_data = sin_sum(input_data).to(device)
    dataset = simpleDataset(input_data, output_data)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    ground_truth_model = two_layer_relu_network(10, 1, 5000).to(device)
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
                         lr = 3e-4, batch_size=1, convergence_req=1e-3, max_epochs=3e4,
                         input_dist=torch.normal, input_dist_args = {"mean":0, "std":1}, normalize_input=True,
                         true_function=lambda x: torch.sin(torch.sum(x, dim=1).unsqueeze(1)),
                         label_noise_sd=0.05, random_seed=137):
    """
    RETURNS: {m: [converged, sharpness, train_loss, valid_loss, mean_training_sparsity, mean_valid_sparsity, loss_curve]}
    ARGS:
    n: int, number of points in training data
    n_valid: int, number of points in validation data
    d: int, dimension of each training data point
    m: list[int], widths of neural network to run experiment on
    lr: float, learning rate
    convergence_req: float, the NN will be considered converged if EVERY data points has loss this low and training will halt
    max_epochs: float, the NN will run for this many epochs before giving up on convergence
    input_dist: torch function, determines distribution of input data
    input_dist_args: dict{str: float}, a dictionary with parameters like mean and sd for the input_dist. Do not include the size parameter.
    normalize_input: bool, whether to normalize input trian and valid data to have norm 1. Useful to convert gaussian data to uniform on the hypersphere.
    true_function: torch function, output = torch_function(input). The function should act on the full nxd matrix of input data
    label_noise_sd: float, gives the standard deviation for gaussian noise in noisy SGD
    random_seed: int
    """

    experiment_results = {}

    # generate data
    torch.manual_seed(random_seed)
    input_data = input_dist(**input_dist_args, size=(n, d)).to(device)
    if normalize_input:
        input_data = input_data / torch.norm(input_data, dim=0)
    output_data = true_function(input_data).to(device)

    valid_input = input_dist(**input_dist_args, size=(n_valid, d)).to(device)
    if normalize_input:
        valid_input = valid_input / torch.norm(valid_input, dim=0)
    valid_output = true_function(valid_input).to(device)

    dataset = simpleDataset(input_data, output_data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    print("Training experimental models")
    # run NN for each width
    for width in m:
        model = two_layer_relu_network(d, 1, width).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss = np.inf
        converged = False
        loss_curve = []
        epochs = max_epochs

        with tqdm(range(int(epochs)), desc="Training Progress, m=" + str(width)) as progress_bar:
            for epoch in progress_bar:

                all_loss = torch.tensor([]).to(device)
                for data, labels in dataloader:

                    optimizer.zero_grad()
                    model_output = model(data)
                    label_noise = torch.normal(0, label_noise_sd, size=(1,)).to(device)       # FIXME: experiment with size of label noise
                    loss = criterion(model_output, labels + label_noise)
                    loss.backward(retain_graph=True)

                    if epoch != epochs-1:
                        optimizer.step()

                    loss_unnoisy = criterion(model_output, labels)
                    all_loss = torch.cat((all_loss, loss_unnoisy.unsqueeze(0)))       # FIXME: don't run inference twice per iter

                if epoch % int(epochs/200) == 0:
                    loss_curve.append(torch.mean(all_loss).item())

                progress_bar.set_postfix(avg_loss=torch.mean(all_loss).item(), max_loss=torch.max(all_loss).item())

                if all(all_loss < convergence_req):
                    model_output = model(data)
                    loss = criterion(model_output, labels)
                    converged = True
                    print("Model with width", width, "interpolated in", epoch, "epochs")
                    break
                else:
                    if epoch == epochs - 1:
                        print("Model with width", width, "did not interpolate")


        def point_sharpness(x):
            gradients = torch.autograd.grad(model(x), model.parameters(), create_graph=True)        # FIXME: confirm this calculation
            return torch.linalg.vector_norm(torch.cat([g.flatten() for g in gradients]))**2
        sharpness = 2/len(dataloader) * np.mean(np.array([point_sharpness(x[0]).item() for x in dataloader]))

        train_loss = criterion(model(input_data), output_data).item()
        valid_loss = criterion(model(valid_input), valid_output).item()
        mean_training_sparsity = torch.mean(torch.Tensor([torch.sum(model.fc1(x) > 0) for x in input_data])) / m
        mean_valid_sparsity = torch.mean(torch.Tensor([torch.sum(model.fc1(x) > 0) for x in valid_input])) / m

        experiment_results[width] = {"converged":converged, "widths":m, "sharpness":sharpness, "train_loss":train_loss, "valid_loss":valid_loss, "mean_training_sparsity":mean_training_sparsity, "mean_valid_sparsity":mean_valid_sparsity, "loss_curve":loss_curve}

    print("Completed training experimental models")
    return experiment_results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Connected to", str(device))
    ground_truth_model = train_true_model()
    results = run_width_experiment(n=100, d=10, m=np.arange(10, 101, 10), true_function=ground_truth_model)
    with open("./experiment_results/exp1.pkl", 'wb') as file:
        pickle.dump(results, file)