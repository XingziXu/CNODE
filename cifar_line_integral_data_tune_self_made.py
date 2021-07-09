from __future__ import print_function
import argparse
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint_adjoint as odeint
from scipy.integrate import odeint as odeint_scipy
from torch.autograd import Variable
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset

class Grad_net(nn.Module):
    def __init__(self, input_size : int, width : int, output_size : int):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size,width),
            nn.ReLU(),
            nn.GroupNorm(1,width),
           # nn.LayerNorm(width),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.GroupNorm(1,width),
           # nn.LayerNorm(width),
            nn.Linear(width,output_size),
            nn.Tanhshrink()
        )

    def forward(self,x):
        y_pred = self.stack(x)
        return y_pred

#model = Grad_net()

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ODEFunc(nn.Module):# define ode function, this is what we train on

    def __init__(self, input_size : int, width : int, output_size : int):
        super(ODEFunc, self).__init__()
        self.l1 = nn.Linear(input_size, width)
        self.l2 = nn.Linear(width,width)
        self.l3 = nn.Linear(width, output_size)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, t):
        t = self.l1(t)
        t = F.relu(t)
        t = self.norm1(t)
        t = self.l2(t)
        t = F.relu(t)
        t = self.norm2(t)
        t = self.l3(t)
        t = F.relu(t)
        return t

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3,8,3,1)
        self.conv2 = nn.Conv2d(8,8,3,1)
        self.fc1 = nn.Linear(1568,128)
        self.fc2 = nn.Linear(128,10)
        self.norm1 = nn.BatchNorm2d(8)
        self.norm2 = nn.BatchNorm2d(8)
        self.norm3 = nn.GroupNorm(8,128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.norm3(x)
        x = self.fc2(x)
        return x

class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0, float('inf'))
            module.weight.data = w

def train_cifar(config, checkpoint_dir=None):
    #net = Net(config["l1"], config["l2"])
    encoder = Encoder()
    input_size_path = 11
    width_path = 64
    output_size_path = 2
    input_size_grad = 14
    #width_grad = 64
    output_size_grad = 12
    clipper = WeightClipper()
    path_net = ODEFunc(input_size_path, width_path, output_size_path)
    path_net.apply(clipper)
    grad_x_net = Grad_net(input_size_grad, config["l1"], output_size_grad)
    grad_y_net = Grad_net(input_size_grad, config["l1"], output_size_grad)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            encoder = nn.DataParallel(encoder)
            path_net = nn.DataParallel(path_net)
            grad_x_net = nn.DataParallel(grad_x_net)
            grad_y_net = nn.DataParallel(grad_y_net)
    encoder.to(device)
    path_net.to(device)
    grad_x_net.to(device)
    grad_y_net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(encoder.parameters())+list(path_net.parameters())+list(grad_x_net.parameters())+list(grad_y_net.parameters()), lr=config["lr"])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        encoder_state, path_net_state, grad_x_net_state, grad_y_net_state, optimizer_state = torch.load(checkpoint)
        encoder.load_state_dict(encoder_state)
        path_net.load_state_dict(path_net_state)
        grad_x_net.load_state_dict(grad_x_net_state)
        grad_y_net.load_state_dict(grad_y_net_state)
        optimizer.load_state_dict(optimizer_state)

    data_dir = os.path.abspath("./data")
    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            dt = ((1.0-0.0)/50.0)
            p_current = encoder(data)
            p_i = p_current
            p_current = torch.cat((p_current,torch.zeros(p_current.size(0),2).to(device)),dim=1) # augment here
            for iter in range(1,int(50.0)+1): # for each random value, integrate from 0 to 1
                t_data_current = torch.cat((iter*dt*torch.ones((p_current.size(0),1)).to(device),p_i),dim=1) # calculate the current time
                t_data_current = Variable(t_data_current.data, requires_grad=True)
                g_h_current = path_net(t_data_current)
                dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1) # calculate the current dg/dt
                dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1)
                in_grad = torch.cat((p_current.view(p_current.size()[0], p_current.size()[1]), g_h_current), dim=1)
                p_current = p_current + dt*(grad_x_net(in_grad)*dg_dt_current + grad_y_net(in_grad)*dh_dt_current)
            soft_max = nn.Softmax(dim=1)
            p_current = p_current[:,0:10] # the first ten are features
            ####### neural path integral ends here #######
            outputs = soft_max(p_current)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                dt = ((1.0-0.0)/50.0)
                p_current = encoder(data)
                p_i = p_current
                p_current = torch.cat((p_current,torch.zeros(p_current.size(0),2).to(device)),dim=1) # augment here
                for iter in range(1,int(50.0)+1): # for each random value, integrate from 0 to 1
                    t_data_current = torch.cat((iter*dt*torch.ones((p_current.size(0),1)).to(device),p_i),dim=1) # calculate the current time
                    t_data_current = Variable(t_data_current.data, requires_grad=True)
                    g_h_current = path_net(t_data_current)
                    dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                    dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1) # calculate the current dg/dt
                    dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                    dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1)
                    in_grad = torch.cat((p_current.view(p_current.size()[0], p_current.size()[1]), g_h_current), dim=1)
                    p_current = p_current + dt*(grad_x_net(in_grad)*dg_dt_current + grad_y_net(in_grad)*dh_dt_current)
                soft_max = nn.Softmax(dim=1)
                p_current = p_current[:,0:10] # the first ten are features
                ####### neural path integral ends here #######
                outputs = soft_max(p_current)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (encoder.state_dict(), path_net.state_dict(), grad_x_net.state_dict(), grad_y_net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

def test_best_model(best_trial):
    input_size_path = 11
    width_path = 64
    output_size_path = 2
    input_size_grad = 14
    #width_grad = 64
    output_size_grad = 12
    encoder = Encoder()
    path_net = ODEFunc(input_size_path, width_path, output_size_path)
    grad_x_net = Grad_net(input_size_grad, best_trial.config["l1"], output_size_grad)
    grad_y_net = Grad_net(input_size_grad, best_trial.config["l1"], output_size_grad)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    encoder.to(device)
    path_net.to(device)
    grad_x_net.to(device)
    grad_y_net.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    encoder_state, path_net_state, grad_x_net_state, grad_y_net_state, optimizer_state = torch.load(checkpoint_path)
    encoder.load_state_dict(encoder_state)
    path_net.load_state_dict(path_net_state)
    grad_x_net.load_state_dict(grad_x_net_state)
    grad_y_net.load_state_dict(grad_y_net_state)

    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            dt = ((1.0-0.0)/50.0)
            p_current = encoder(images)
            p_i = p_current
            p_current = torch.cat((p_current,torch.zeros(p_current.size(0),2).to(device)),dim=1) # augment here
            for iter in range(1,int(50.0)+1): # for each random value, integrate from 0 to 1
                t_data_current = torch.cat((iter*dt*torch.ones((p_current.size(0),1)).to(device),p_i),dim=1) # calculate the current time
                t_data_current = Variable(t_data_current.data, requires_grad=True)
                g_h_current = path_net(t_data_current)
                dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1) # calculate the current dg/dt
                dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1)
                in_grad = torch.cat((p_current.view(p_current.size()[0], p_current.size()[1]), g_h_current), dim=1)
                p_current = p_current + dt*(grad_x_net(in_grad)*dg_dt_current + grad_y_net(in_grad)*dh_dt_current)
            soft_max = nn.Softmax(dim=1)
            p_current = p_current[:,0:10] # the first ten are features
            ####### neural path integral ends here #######
            outputs = soft_max(p_current)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("Best trial test set accuracy: {}".format(correct / total))



def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "l1": tune.sample_from(lambda _: 32 * np.random.randint(1, 4)),
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    result = tune.run(
        tune.with_parameters(train_cifar),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    if ray.util.client.ray.is_connected():
        # If using Ray Client, we want to make sure checkpoint access
        # happens on the server. So we wrap `test_best_model` in a Ray task.
        # We have to make sure it gets executed on the same node that
        # ``tune.run`` is called on.
        from ray.tune.utils.util import force_on_current_node
        remote_fn = force_on_current_node(ray.remote(test_best_model))
        ray.get(remote_fn.remote(best_trial))
    else:
        test_best_model(best_trial)

if __name__ == '__main__':
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=2)
