# Importing required libraries
import torch as pt 
from torch.utils.data import DataLoader
from torch import nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import time


# Defining our neural network class as per given instruction in the assignment
# Here in our conv1 layer 28x28x1 goes to 28x28x8 and for conv2 14x14x8 -> 12x12x16 
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (3,3), padding = 'same')
        self.conv2 = nn.Conv2d(6, 16, (3,3))
        self.fc1   = nn.Linear(16*3*3, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, 10)
        self.max_pool = nn.MaxPool2d((2,2))
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

def train_func(model, dataloader, rank, epochs = 20, epsi = 10e-5):
    fn_loss = nn.CrossEntropyLoss()
    optm = pt.optim.SGD(model.parameters(), lr = 0.1) 
    size = len(dataloader.dataset)
    for i in range(epochs):
        avg_loss = 0
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


def eval_func(model,test_dataloader,fn_loss):
    size = len(test_dataloader.dataset)
    ts_loss, accu = 0, 0
    with pt.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            ts_loss = ts_loss + (fn_loss(pred, y).item() * X.size(0))
            accu = accu + (pred.argmax(1) == y).type(pt.float).sum().item()
    ts_loss = ts_loss / size
    accu = accu / size
    print(f"Test Accu: {(100 * accu):>0.1f}%, Avg loss: {ts_loss:>8f}")
    return ts_loss, 100 * accu


def splt_func(dataset,size):
    splt_func = [len(dataset.targets)//size] * size
    dataset = [i for i in pt.utils.data.random_split(dataset, splt_func)]
    return dataset

# Most prominent changes can be seen in this main function of our part since I took
# the approach of splitting the data and giving it to workers where all the workers different function
# until convergence happens. So when we are not using MPI we have changes in following main part.
# As we can see there is no more handling of root worker specific task in a manner as we were doing in MPI.
# We dont have to maintain some sort of state dictionary as its been taken care of with shared memory.
# This loop will run for all my 4 number of workers. So in one execution we have output for all different variations.
if __name__ == '__main__':
    for size in range(1, 5):
        s_time = time.time()
        transform_list = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        MNIST_train = MNIST(root = ".", train = True, download = True, transform = transform_list)
        MNIST_test = MNIST(root = ".", train = False, download = True, transform = transform_list)
        test_dataloader = DataLoader(MNIST_test, batch_size = 1000, shuffle = False)
		
        if size > 1:
            MNIST_train = splt_func(MNIST_train, size)
        else:
            MNIST_train = [MNIST_train]
        model = ConvNet()
        model.share_memory()
        processes = []
        for rank in range(size):
            train_dataloader = DataLoader(MNIST_train[rank], batch_size=32, shuffle=True)
            p = mp.Process(target = train_func, args = (model, train_dataloader, rank))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        fn_loss = nn.CrossEntropyLoss()
        eval_func(model,test_dataloader, fn_loss) 
        print(f'Total time with {size} workers : {time.time() - s_time}')

    
