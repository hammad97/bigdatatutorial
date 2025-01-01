# Importing required libraries
import numpy as np
import torch as pt 
from torch.utils.data import DataLoader
from torch import nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import time
from mpi4py import MPI
import warnings

# Setting up initial variables for MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
s_time = time.time()
warnings.filterwarnings("ignore", category = DeprecationWarning) 


# Defining our neural network class as per given instruction in the assignment
# Here in our conv1 layer 28x28x1 goes to 28x28x8 and for conv2 14x14x8 -> 12x12x16 
# Similar definition has been used as implemented in tutorial 7
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (3,3), padding = 'same') 
        self.conv2 = nn.Conv2d(6, 16, (3,3))
        self.fc1   = nn.Linear(16*3*3, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, 10)
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.max_pool(x)    
        x = x.view(-1, 16*3*3)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


# This is our training function where accepts different parameters from main function
# And then we train it in batches. Firstly by computing prediction and its loss and then
# backpropagate meanwhile keeping the avg_loss and later once a batch is finished we update 
# our average loss. I have also included another stopping criteria as it was discussed in last lab.
def train_func(model, optm, fn_loss, dataloader, epochs, rank = 0, epsi = 10e-5):
    size = len(dataloader.dataset)
    print(f'(Worker : {rank} , Size : {size}) Training.')
    for i in range(epochs):
        avg_loss = 0
        accu = 0
        for batch, (X, y) in enumerate(dataloader):
            optm.zero_grad()
            l_val = fn_loss(model(X), y)
            l_val.backward()
            optm.step()
            avg_loss = avg_loss + (l_val.item() * X.size(0))
        avg_loss = avg_loss / size
        print(f'Worker : {rank}, Iteration : {i}, Loss : {avg_loss}.')
        if i != 0:
            if avg_loss_last - avg_loss < epsi:
                print(f'Convergence *** Worker : {rank}, Iteration : {i}')
                break            
        avg_loss_last = avg_loss 
    return

# This function does the evaluation where no_grad helps in saving time and memory and doesnt affect accuracy.
# So perform our evaluation with the test dataset and check for the loss and accuracy.
# then we print them in output as mentioned in question.
def eval_func(model, test_dataloader, fn_loss):
    size = len(test_dataloader.dataset)
    ts_loss, accu = 0, 0
    with pt.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            ts_loss = ts_loss + (fn_loss(pred, y).item() * X.size(0))
            accu = accu + (pred.argmax(1) == y).type(pt.float).sum().item()
    ts_loss = ts_loss / size
    accu = accu  / size
    print(f"Test Accu: {(100 * accu):>0.1f}%, Avg loss: {ts_loss:>8f}")
    return ts_loss, 100 * accu


# This is our splitting function where I have just performed a random split and returned the updated dataset.
def splt_func(dataset, size):
    len_datasets = [len(dataset.targets)//size] * size
    dataset = [tmp for tmp in pt.utils.data.random_split(dataset, len_datasets)]
    return dataset

# In our main function first worker0 downloads the MNIST dataset and splits it.
# Then we send the splitted dataset to other workers, then worker0 extract model_weights and update our states_dictionary
# And all the other workers load state from the given state dictionary updated by worker0
# Then we invoke our training function and later perform our evaluation function
# And in the end print the accuracy and loss for with and without averaging the weights.
if rank == 0:
    MNIST_train = MNIST(root = ".", train = True, download = True, transform = transforms.ToTensor())
    MNIST_test = MNIST(root = ".", train = False, download = True, transform = transforms.ToTensor())
    test_dataloader = DataLoader(MNIST_test, batch_size = 1000, shuffle = False)
    if size > 1:
        MNIST_train = splt_func(MNIST_train, size)
else:
    MNIST_train = None
if size > 1:
    MNIST_train = comm.scatter(MNIST_train, root = 0)

train_dataloader = DataLoader(MNIST_train, batch_size = 32, shuffle = True)
model = ConvNet()
state_dict = {}
if rank == 0:
    for name, param in model.named_parameters():
        state_dict[name] = param.detach().numpy()
state_dicts = comm.bcast(state_dict, root = 0)
if rank != 0:
    for name, param in model.named_parameters():
        model.state_dict()[name][:] = pt.Tensor(state_dicts[name])
fn_loss = nn.CrossEntropyLoss()
opt = pt.optim.SGD(model.parameters(), lr = 0.1)
train_func(model, opt, fn_loss, train_dataloader, epochs = 20, rank = rank)
state_dict = {}
for name, param in model.named_parameters():
    state_dict[name] = param.detach().numpy() 
state_dicts = comm.gather(state_dict, root = 0)
if rank == 0:
    print('Model Evaluation')
    eval_func(model,test_dataloader,fn_loss)
    if size > 1:
        print('Model evaluation after weights averaged on Worker 0')
        for name, param in model.named_parameters():
            mean_weights = np.mean([tmp[name][np.newaxis,...] for tmp in state_dicts], axis = 0)
            model.state_dict()[name][:] = pt.Tensor(mean_weights)
        eval_func(model, test_dataloader, fn_loss)
print(f'Worker : {rank} took time : {time.time() - s_time:.2f}')