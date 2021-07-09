# -*- coding: utf-8 -*-
"""
Hyperparameter tuning with Ray Tune
===================================
Hyperparameter tuning can make the difference between an average model and a highly
accurate one. Often simple things like choosing a different learning rate or changing
a network layer size can have a dramatic impact on your model performance.
Fortunately, there are tools that help with finding the best combination of parameters.
`Ray Tune <https://docs.ray.io/en/latest/tune.html>`_ is an industry standard tool for
distributed hyperparameter tuning. Ray Tune includes the latest hyperparameter search
algorithms, integrates with TensorBoard and other analysis libraries, and natively
supports distributed training through `Ray's distributed machine learning engine
<https://ray.io/>`_.
In this tutorial, we will show you how to integrate Ray Tune into your PyTorch
training workflow. We will extend `this tutorial from the PyTorch documentation
<https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_ for training
a CIFAR10 image classifier.
As you will see, we only need to add some slight modifications. In particular, we
need to
1. wrap data loading and training in functions,
2. make some network parameters configurable,
3. add checkpointing (optional),
4. and define the search space for the model tuning
|
To run this tutorial, please make sure the following packages are
installed:
-  ``ray[tune]``: Distributed hyperparameter tuning library
-  ``torchvision``: For the data transformers
Setup / Imports
---------------
Let's start with the imports:
"""
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.autograd import Variable

######################################################################
# Most of the imports are needed for building the PyTorch model. Only the last three
# imports are for Ray Tune.
#
# Data loaders
# ------------
# We wrap the data loaders in their own function and pass a global data directory.
# This way we can share a data directory between different trials.


def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset

######################################################################
# Configurable neural network
# ---------------------------
# We can only tune those parameters that are configurable. In this example, we can specify
# the layer sizes of the fully connected layers:

class Grad_net(nn.Module):
    def __init__(self, input_size : int, width_grad : int, output_size : int):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size,width_grad),
            nn.ReLU(),
            nn.GroupNorm(1,width_grad),
           # nn.LayerNorm(width),
            nn.Linear(width_grad,width_grad),
            nn.ReLU(),
            nn.GroupNorm(1,width_grad),
           # nn.LayerNorm(width),
            nn.Linear(width_grad,output_size),
            nn.Tanhshrink()
        )

    def forward(self,x):
        y_pred = self.stack(x)
        return y_pred

class ODEFunc(nn.Module):# define ode function, this is what we train on

    def __init__(self, input_size : int, width_path : int, output_size : int):
        super(ODEFunc, self).__init__()
        self.l1 = nn.Linear(input_size, width_path)
        self.l2 = nn.Linear(width_path,width_path)
        self.l3 = nn.Linear(width_path, output_size)
        self.norm1 = nn.LayerNorm(width_path)
        self.norm2 = nn.LayerNorm(width_path)

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
######################################################################
# The train function
# ------------------
# Now it gets interesting, because we introduce some changes to the example `from the PyTorch
# documentation <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_.
#
# We wrap the training script in a function ``train_cifar(config, checkpoint_dir=None, data_dir=None)``.
# As you can guess, the ``config`` parameter will receive the hyperparameters we would like to
# train with. The ``checkpoint_dir`` parameter is used to restore checkpoints. The ``data_dir`` specifies
# the directory where we load and store the data, so multiple runs can share the same data source.
#
# .. code-block:: python
#
#     net = Net(config["l1"], config["l2"])
#
#     if checkpoint_dir:
#         model_state, optimizer_state = torch.load(
#             os.path.join(checkpoint_dir, "checkpoint"))
#         net.load_state_dict(model_state)
#         optimizer.load_state_dict(optimizer_state)
#
# The learning rate of the optimizer is made configurable, too:
#
# .. code-block:: python
#
#     optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
#
# We also split the training data into a training and validation subset. We thus train on
# 80% of the data and calculate the validation loss on the remaining 20%. The batch sizes
# with which we iterate through the training and test sets are configurable as well.
#
# Adding (multi) GPU support with DataParallel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Image classification benefits largely from GPUs. Luckily, we can continue to use
# PyTorch's abstractions in Ray Tune. Thus, we can wrap our model in ``nn.DataParallel``
# to support data parallel training on multiple GPUs:
#
# .. code-block:: python
#
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             net = nn.DataParallel(net)
#     net.to(device)
#
# By using a ``device`` variable we make sure that training also works when we have
# no GPUs available. PyTorch requires us to send our data to the GPU memory explicitly,
# like this:
#
# .. code-block:: python
#
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#
# The code now supports training on CPUs, on a single GPU, and on multiple GPUs. Notably, Ray
# also supports `fractional GPUs <https://docs.ray.io/en/master/using-ray-with-gpus.html#fractional-gpus>`_
# so we can share GPUs among trials, as long as the model still fits on the GPU memory. We'll come back
# to that later.
#
# Communicating with Ray Tune
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The most interesting part is the communication with Ray Tune:
#
# .. code-block:: python
#
#     with tune.checkpoint_dir(epoch) as checkpoint_dir:
#         path = os.path.join(checkpoint_dir, "checkpoint")
#         torch.save((net.state_dict(), optimizer.state_dict()), path)
#
#     tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
#
# Here we first save a checkpoint and then report some metrics back to Ray Tune. Specifically,
# we send the validation loss and accuracy back to Ray Tune. Ray Tune can then use these metrics
# to decide which hyperparameter configuration lead to the best results. These metrics
# can also be used to stop bad performing trials early in order to avoid wasting
# resources on those trials.
#
# The checkpoint saving is optional, however, it is necessary if we wanted to use advanced
# schedulers like
# `Population Based Training <https://docs.ray.io/en/master/tune/tutorials/tune-advanced-tutorial.html>`_.
# Also, by saving the checkpoint we can later load the trained models and validate them
# on a test set.
#
# Full training function
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The full code example looks like this:


def train_cifar(config, checkpoint_dir=None, data_dir=None):
    encoder = Encoder()
    input_size_path = 11
    width_path =64
    output_size_path = 2
    input_size_grad = 12
    width_grad = 64
    output_size_grad = 10
    clipper = WeightClipper()
    path_net = ODEFunc(input_size_path, width_path, output_size_path)
    path_net.apply(clipper)
    grad_x_net = Grad_net(input_size_grad, width_grad, output_size_grad)
    grad_y_net = Grad_net(input_size_grad, width_grad, output_size_grad)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
#        if torch.cuda.device_count() > 1:
#            encoder = nn.DataParallel(encoder)
#            path_net = nn.DataParallel(path_net)
#            grad_x_net = nn.DataParallel(grad_x_net)
#            grad_y_net = nn.DataParallel(grad_y_net)
    encoder.to(device)
    path_net.to(device)
    grad_x_net.to(device)
    grad_y_net.to(device)

    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(encoder.parameters())+list(path_net.parameters())+list(grad_x_net.parameters())+list(grad_y_net.parameters()), lr=config["lr"])

    if checkpoint_dir:
        optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        #path_net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=1)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=1)

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
            dt = ((1.-0.)/1e2)
            p_current = encoder(data)
            p_i = p_current
            for iter in range(1,int(1e2)+1): # for each random value, integrate from 0 to 1
                print("we are training")
                t_data_current = torch.cat((iter*dt*torch.ones((p_current.size(0),1)).to(device),p_i),dim=1) # calculate the current time
                t_data_current = Variable(t_data_current.data, requires_grad=True)
                g_h_current = path_net(t_data_current)
                dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1) # calculate the current dg/dt
                dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1)
                in_grad = torch.cat((p_current.view(p_current.size()[0], 10), g_h_current), dim=1)
                p_current = p_current + dt*(grad_x_net(in_grad)*dg_dt_current + grad_y_net(in_grad)*dh_dt_current)
            soft_max = nn.Softmax(dim=1)
            ####### neural path integral ends here #######
            outputs = soft_max(p_current)
            loss = F.cross_entropy(p_current, labels)
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

                dt = ((1.-0.)/1e2)
                p_current = encoder(data)
                p_i = p_current
                for iter in range(1,int(1e2)+1): # for each random value, integrate from 0 to 1
                    print("we are validating")
                    t_data_current = torch.cat((iter*dt*torch.ones((p_current.size(0),1)).to(device),p_i),dim=1) # calculate the current time
                    t_data_current = Variable(t_data_current.data, requires_grad=True)
                    g_h_current = path_net(t_data_current)
                    dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                    dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1) # calculate the current dg/dt
                    dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                    dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1)
                    in_grad = torch.cat((p_current.view(p_current.size()[0], 10), g_h_current), dim=1)
                    p_current = p_current + dt*(grad_x_net(in_grad)*dg_dt_current + grad_y_net(in_grad)*dh_dt_current)
                soft_max = nn.Softmax(dim=1)
                ####### neural path integral ends here #######
                outputs = soft_max(p_current)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = F.cross_entropy(p_current, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((path_net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

######################################################################
# As you can see, most of the code is adapted directly from the original example.
#
# Test set accuracy
# -----------------
# Commonly the performance of a machine learning model is tested on a hold-out test
# set with data that has not been used for training the model. We also wrap this in a
# function:


def test_accuracy(encoder, path_net, grad_x_net, grad_y_net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=1)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            dt = ((1.-0.)/1e2)
            p_current = encoder(images)
            p_i = p_current
            for iter in range(1,int(1e2)+1): # for each random value, integrate from 0 to 1
                print("we are testing")
                t_data_current = torch.cat((iter*dt*torch.ones((p_current.size(0),1)).to(device),p_i),dim=1) # calculate the current time
                t_data_current = Variable(t_data_current.data, requires_grad=True)
                g_h_current = path_net(t_data_current)
                dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1) # calculate the current dg/dt
                dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
                dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1)
                in_grad = torch.cat((p_current.view(p_current.size()[0], 10), g_h_current), dim=1)
                p_current = p_current + dt*(grad_x_net(in_grad)*dg_dt_current + grad_y_net(in_grad)*dh_dt_current)
            soft_max = nn.Softmax(dim=1)
            ####### neural path integral ends here #######
            outputs = soft_max(p_current)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

######################################################################
# The function also expects a ``device`` parameter, so we can do the
# test set validation on a GPU.
#
# Configuring the search space
# ----------------------------
# Lastly, we need to define Ray Tune's search space. Here is an example:
#
# .. code-block:: python
#
#     config = {
#         "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([2, 4, 8, 16])
#     }
#
# The ``tune.sample_from()`` function makes it possible to define your own sample
# methods to obtain hyperparameters. In this example, the ``l1`` and ``l2`` parameters
# should be powers of 2 between 4 and 256, so either 4, 8, 16, 32, 64, 128, or 256.
# The ``lr`` (learning rate) should be uniformly sampled between 0.0001 and 0.1. Lastly,
# the batch size is a choice between 2, 4, 8, and 16.
#
# At each trial, Ray Tune will now randomly sample a combination of parameters from these
# search spaces. It will then train a number of models in parallel and find the best
# performing one among these. We also use the ``ASHAScheduler`` which will terminate bad
# performing trials early.
#
# We wrap the ``train_cifar`` function with ``functools.partial`` to set the constant
# ``data_dir`` parameter. We can also tell Ray Tune what resources should be
# available for each trial:
#
# .. code-block:: python
#
#     gpus_per_trial = 2
#     # ...
#     result = tune.run(
#         partial(train_cifar, data_dir=data_dir),
#         resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         checkpoint_at_end=True)
#
# You can specify the number of CPUs, which are then available e.g.
# to increase the ``num_workers`` of the PyTorch ``DataLoader`` instances. The selected
# number of GPUs are made visible to PyTorch in each trial. Trials do not have access to
# GPUs that haven't been requested for them - so you don't have to care about two trials
# using the same set of resources.
#
# Here we can also specify fractional GPUs, so something like ``gpus_per_trial=0.5`` is
# completely valid. The trials will then share GPUs among each other.
# You just have to make sure that the models still fit in the GPU memory.
#
# After training the models, we will find the best performing one and load the trained
# network from the checkpoint file. We then obtain the test set accuracy and report
# everything by printing.
#
# The full main function looks like this:


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        #"l1": tune.sample_from(lambda _: 2 ** np.random.randint(16, 64)),
        #"l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128])
    }
    ray.init(log_to_driver=False)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    encoder = Encoder()
    input_size_path = 11
    width_path =64
    output_size_path = 2
    input_size_grad = 12
    width_grad = 64
    output_size_grad = 10
    clipper = WeightClipper()
    grad_x_net = Grad_net(input_size_grad, width_grad, output_size_grad)
    grad_y_net = Grad_net(input_size_grad, width_grad, output_size_grad)
    path_net = ODEFunc(input_size_path, width_path, output_size_path)
    path_net.apply(clipper)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    encoder.to(device)
    grad_x_net.to(device)
    grad_y_net.to(device)
    path_net.to(device)
    

    best_checkpoint_dir = best_trial.checkpoint.value
    optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    #path_net.load_state_dict(model_state)

    test_acc = test_accuracy(encoder, path_net, grad_x_net, grad_y_net, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)


######################################################################
# If you run the code, an example output could look like this:
#
# ::
#
#     Number of trials: 10 (10 TERMINATED)
#     +-----+------+------+-------------+--------------+---------+------------+--------------------+
#     | ... |   l1 |   l2 |          lr |   batch_size |    loss |   accuracy | training_iteration |
#     |-----+------+------+-------------+--------------+---------+------------+--------------------|
#     | ... |   64 |    4 | 0.00011629  |            2 | 1.87273 |     0.244  |                  2 |
#     | ... |   32 |   64 | 0.000339763 |            8 | 1.23603 |     0.567  |                  8 |
#     | ... |    8 |   16 | 0.00276249  |           16 | 1.1815  |     0.5836 |                 10 |
#     | ... |    4 |   64 | 0.000648721 |            4 | 1.31131 |     0.5224 |                  8 |
#     | ... |   32 |   16 | 0.000340753 |            8 | 1.26454 |     0.5444 |                  8 |
#     | ... |    8 |    4 | 0.000699775 |            8 | 1.99594 |     0.1983 |                  2 |
#     | ... |  256 |    8 | 0.0839654   |           16 | 2.3119  |     0.0993 |                  1 |
#     | ... |   16 |  128 | 0.0758154   |           16 | 2.33575 |     0.1327 |                  1 |
#     | ... |   16 |    8 | 0.0763312   |           16 | 2.31129 |     0.1042 |                  4 |
#     | ... |  128 |   16 | 0.000124903 |            4 | 2.26917 |     0.1945 |                  1 |
#     +-----+------+------+-------------+--------------+---------+------------+--------------------+
#
#
#     Best trial config: {'l1': 8, 'l2': 16, 'lr': 0.00276249, 'batch_size': 16, 'data_dir': '...'}
#     Best trial final validation loss: 1.181501
#     Best trial final validation accuracy: 0.5836
#     Best trial test set accuracy: 0.5806
#
# Most trials have been stopped early in order to avoid wasting resources.
# The best performing trial achieved a validation accuracy of about 58%, which could
# be confirmed on the test set.
#
# So that's it! You can now tune the parameters of your PyTorch models.