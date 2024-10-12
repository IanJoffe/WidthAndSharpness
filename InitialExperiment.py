import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from sam import SAM
from grokfast import gradfilter_ma, gradfilter_ema
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
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
    def __init__(self, input_size, output_size, width, scale_init_factor=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, width, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(width, output_size, bias=False)
        if scale_init_factor != 1:
            self.scale_init(scale_init_factor)

    def scale_init(self, factor):
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        self.fc1.weight.data *= factor
        nn.init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        self.fc2.weight.data *= factor


    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        return z2
  
def train_model(width, d,
                device, dataloader, input_data, output_data, valid_input, valid_output, scale_init_factor,
                lr, momentum, use_sam, use_grokfast, grokfast_params, convergence_req, convergence_halt, max_epochs, num_measurements,
                input_dist, input_dist_args, label_noise_dist, label_noise_dist_args, weights_ema, last_epochs_noiseless, noiseless_lr,
                experiment_results_output, model_checkpoints_output, experiment_results_lock, model_checkpoints_lock, random_seed,
                no_wandb, wandb_notes, wandb_storage_location):
    # see run_width_experiment for explanation of parameters

    if not no_wandb:
        wandb.init(
            project = "WidthAndSharpness",
            notes = f'Stored in {wandb_storage_location}. {wandb_notes}'
        )
        wandb.config = {"wdith":width, "d":d, "lr":lr, "momentum":momentum, "use_sam":use_sam, "use_grokfast":use_grokfast,
                        **{f'grokfast_params_{key}':value for key, value in grokfast_params.items()}, "convergence_req":convergence_req,
                        "convergence_halt":convergence_halt, "max_epohcs":max_epochs, "input_dist":input_dist, **{f'input_dist_args_{key}':value for key, value in input_dist_args.items()},
                        "label_noise_dist":label_noise_dist, **{f'label_noise_dist_args_{key}':value for key, value in label_noise_dist_args.items()}, "weights_ema":weights_ema,
                        "last_epochs_noiseless":last_epochs_noiseless, "noiseless_lr":noiseless_lr, "random_seed":random_seed}

    torch.manual_seed(random_seed)

    # set up training
    model = two_layer_relu_network(d, 1, width, scale_init_factor).to(device)
    model_ema = copy.deepcopy(model)
    criterion = nn.MSELoss(reduction="mean")
    if use_sam:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = np.inf
    converged = False
    epochs = max_epochs

    with model_checkpoints_lock:
        model_checkpoints_output[int(width)] = dict()
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
    def average_models(model1, model2, factor):
        averaged_model = two_layer_relu_network(model1.fc1.in_features, model1.fc2.out_features, model1.fc1.out_features)
        for param1, param2, param_avg in zip(model1.parameters(), model2.parameters(), averaged_model.parameters()):
            param_avg.data = factor * param1.data + (1 - factor) * param2.data
        return averaged_model

    with tqdm(range(int(epochs)+1), desc="Training Progress, m=" + str(width)) as progress_bar:
        for epoch in progress_bar:

            # take measurments
            if epoch % int(epochs/num_measurements) == 0:
                if not weights_ema or epoch == 0:
                    measurement_model = model
                else:
                    measurement_model = model_ema
                measured_epochs_measurement = epoch
                train_loss_measurement = criterion(measurement_model(input_data), output_data).item()
                valid_loss_measurement = criterion(measurement_model(valid_input), valid_output).item()
                train_accuracy_measurement = torch.isclose(measurement_model(input_data), output_data, atol=convergence_req).float().mean().item()
                valid_accuracy_measurement = torch.isclose(measurement_model(valid_input), valid_output, atol=convergence_req).float().mean().item()
                sharpness_measurement = get_sharpness(input_data, measurement_model)
                measured_epochs.append(measured_epochs_measurement)
                train_loss.append(train_loss_measurement)
                valid_loss.append(valid_loss_measurement)
                train_accuracy.append(train_accuracy_measurement)
                valid_accuracy.append(valid_accuracy_measurement)
                sharpness.append(sharpness_measurement)    # causes warning and won't run on cuda only for first epoch
                if not no_wandb:
                    wandb.log({"measured_epochs": measured_epochs_measurement, "train_loss":train_loss_measurement, "valid_loss":valid_loss_measurement,
                               "train_accuracy":train_accuracy_measurement, "valid_accuracy":valid_accuracy_measurement, "sharpness":sharpness_measurement})
                with model_checkpoints_lock:
                    model_checkpoints_output_width = model_checkpoints_output[int(width)]
                    model_checkpoints_output_width[epoch] = copy.deepcopy(model).to(torch.device('cpu'))
                    model_checkpoints_output[int(width)] = model_checkpoints_output_width
                
            all_loss = torch.tensor([]).to(device)
            for data, labels in dataloader:

                # take training step
                model_output = model(data)
                if epoch == epochs+1 - last_epochs_noiseless:        # +1 is necessary so that this does not occur in the final epoch when last_epochs_noiseless=0
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = noiseless_lr
                if epoch <= epochs+1 - last_epochs_noiseless:
                    label_noise = label_noise_dist(**label_noise_dist_args, size=(dataloader.batch_sampler.batch_size,)).unsqueeze(dim=1).to(device)
                else:
                    label_noise = torch.zeros(size=(dataloader.batch_sampler.batch_size,)).unsqueeze(dim=1).to(device)
                if use_grokfast:
                    grads = None
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
                    elif use_grokfast and epoch > 20:
                        loss.backward()
                        grads = gradfilter_ema(model, grads=grads, **grokfast_params)
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                loss_unnoisy = criterion(model_output, labels)
                all_loss = torch.cat((all_loss, loss_unnoisy.unsqueeze(0)))

                if weights_ema:
                    model_ema = average_models(model_ema, model, weights_ema)

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

    with experiment_results_lock:
        experiment_results_output[int(width)] = {"converged":converged, "epochs":measured_epochs, "train_loss":train_loss, "valid_loss":valid_loss, "train_accuracy":train_accuracy, "valid_accuracy":valid_accuracy, "sharpness": sharpness}
    return

def train_ground_truth_model(d):
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

    return train_ground_truth_model
  
  
def run_width_experiment(n=100, n_valid=1000, d=10, m=list(range(10, 100, 5)),
                         scale_init_factor=1, lr=3e-4, momentum=0, batch_size=1, use_sam=False, use_grokfast=False, grokfast_params={},
                         convergence_req=1e-3, convergence_halt=False, max_epochs=3e4, num_measurements=200,
                         input_dist=torch.normal, input_dist_args = {"mean":0, "std":1}, normalize_input=True, shuffle_data=False,
                         true_function=lambda x: torch.sin(torch.sum(x, dim=1).unsqueeze(1)),
                         label_noise_dist=torch.zeros, label_noise_dist_args={}, weights_ema=None, last_epochs_noiseless=0, noiseless_lr=3e-4, random_seed=137):
    """
    RETURNS: {m: model}, {m: [converged, sharpness, train_loss, valid_loss, mean_training_sparsity, mean_valid_sparsity, loss_curve]}
    ARGS:
    n: int, number of points in training data
    n_valid: int, number of points in validation data
    d: int, dimension of each training data point
    m: list[int], widths of neural network to run experiment on
    scale_init_factor: float, adjust the bounds of the Kaiming initialization's uniform distribution by this factor
    lr: float, learning rate
    momentum: float, momentum parameter for SGD
    use_sam: bool, set to true to use the sam optimizer, if false will use SGD with label noise
    use_grokfast: bool, set to true to apply grokfast https://arxiv.org/pdf/2405.20233
    grokfast_params: dict, paramters to call gradfilter_ema with. can contain alpha and lam. 
    convergence_req: float, the NN will be considered converged if EVERY data points has loss this low, also used to calculate accuracy
    convergence_halt: bool, set to True to halt training once converged
    max_epochs: int, the NN will run for this many epochs before giving up on convergence
    num_measurements: int, validation loss and sharpness will be measured very max_epochs/num_measurements epochs
    input_dist: torch function, determines distribution of input data
    input_dist_args: dict{str: float}, a dictionary with parameters like mean and sd for the input_dist. Do not include the size parameter.
    normalize_input: bool, whether to normalize input trian and valid data to have norm 1. Useful to convert gaussian data to uniform on the hypersphere.
    shuffle_data: whether to shuffle the order of the (data, label) pairs before each epoch of SGD
    true_function: function, output = torch_function(input). The function should act on the full nxd matrix of input data
    label_noise_dist: func, a function that outputs random label noise. Defaults to zero label noise. Must include a `size ` parameter.
    label_noise_dist_args: dict, The arguments to pass into the label_noise_dist
    weights_ema: float or None, If float the model used to calculate loss (but not the model that will be gradient-updated) is an ema of previous models, where the last model has weights_ema weight
    last_epochs_noiseless: int, the last last_epochs_noiseless epochs will have 0 label noise
    noiseless_lr: float, the learning rate during the final noiseless epochs
    random_seed: int
    """

    model_parameters = {"n": n, "d": d, "lr":lr, "label_noise_sd":(label_noise_dist_args["sd"] if "sd" in label_noise_dist_args else "NA"), "max_epochs":max_epochs, "weights_ema":weights_ema, "last_epochs_noiseless":last_epochs_noiseless}

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

    # manage parallelism
    processes = []
    manager = mp.Manager()
    experiment_results = manager.dict()
    model_checkpoints = manager.dict()
    experiment_results_lock = manager.Lock()
    model_checkpoints_lock = manager.Lock()

    print("Training experimental models")
    for width in m:
        p = mp.Process(target=train_model, args=(
            width, d,
            device, dataloader, input_data, output_data, valid_input, valid_output, scale_init_factor,
            lr, momentum, use_sam, use_grokfast, grokfast_params, convergence_req, convergence_halt, max_epochs, num_measurements,
            input_dist, input_dist_args, label_noise_dist, label_noise_dist_args, weights_ema, last_epochs_noiseless, noiseless_lr,
            experiment_results, model_checkpoints, experiment_results_lock, model_checkpoints_lock, random_seed+(width-m[0]),
            no_wandb, wandb_notes, wandb_storage_location
        ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("Completed training experimental models")
    wandb.finish()
    return model_parameters, model_checkpoints, experiment_results

def xor(x):
    return (x[:, 1] * x[:, 2]).unsqueeze(1)

def decaying_pairprod(x, alpha=2):
    if x.shape[1] % 2 == 0:
        return torch.sum(x[:,::2] * x[:,1::2] / (torch.arange(1,x.shape[1]//2 + 1, device=x.device) ** alpha), dim=1).unsqueeze(1)
    else:
        return torch.sum(x[:,::2] *  torch.cat((x[:,1::2], torch.ones(x.shape[0]).unsqueeze(1)), dim=1).to(x.device) / (torch.arange(1,x.shape[1]//2, device=x.device) ** alpha), dim=1).unsqueeze(1)

def sinsum(x):
    return torch.sin(torch.sum(x, dim=1).unsqueeze(1))

def torch_binary_input(size):
    return torch.randint(0, 2, size=size, dtype=torch.float32)*2-1

def torch_binary_label_noise(sd, size):
    return torch.randint(0, 2, size=size, dtype=torch.float32)*2*sd-sd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help="The path to the json file to save results to")
    parser.add_argument("--experiment_text", type=str, help="A call to run_width_experiment() to run")
    parser.add_argument("--wandb_notes", type=str, help="Notes to log in the Weights & Biases description")
    parser.add_argument("-f", "--no_wandb", action="store_true", help="Use this flag to not log the experiment on Weights & Biases")
    
    args = parser.parse_args()
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)

    wandb_storage_location = args.results_file
    wandb_notes = args.wandb_notes
    no_wandb = args.no_wandb

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.init() 
    print("Connected to", str(device))
    mp.set_start_method("spawn")    # may want to set force=True if issues

    # ground_truth_model = train_ground_truth_model(d=30)
    if args.experiment_text is None:
        model_parameters, trained_models, results = run_width_experiment()
    else:
        model_parameters, trained_models, results = eval(args.experiment_text)    # make sure to use single quotes, not double, if making a string within experiment_text

    for m in trained_models.keys():
        (Path(args.results_file).parent / ("checkpoints_" + str(m))).mkdir()
        for epoch in trained_models[m].keys():
            torch.save(trained_models[m][epoch], Path(args.results_file).parent / ("checkpoints_" + str(m)) / ("model_" + str(m) + "_epoch_" + str(epoch) + ".pth"))
    with open(args.results_file, 'w') as f:
        json.dump(dict(results), f)
    with open(Path(args.results_file).parent / "model_params.json", 'w') as f:
        json.dump(model_parameters, f)