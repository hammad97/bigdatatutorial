
## Exercise Sheet 4

# Distributed Machine Learning (Supervised)

In this exercise sheet you are going to implement a supervised machine learning algorithm in a distributed
setting using MPI (mpi4py). We will pick a simple Linear Regression model and train it using a parallel
stochastic gradient algorithm (PSGD)^1.

## 0.1 Dataset

You have to use two datasets for this task. Read the description of the dataset. You have to use them
for learning a regression model.

- Dynamic Features of VirusShare Executables Data Sethttps://archive.ics.uci.edu/ml/datasets/
    Dynamic+Features+of+VirusShare+Executables
- KDD Cup 1998 Data Data Sethttps://archive.ics.uci.edu/ml/datasets/KDD+Cup+1998+
    Data

Please divide your data into 70% train and 30% test. You will train your model on train part of the data
and will test it on test part of the data. Use RMSE train and test score for measuring the performance
of your model.

## 0.2 Linear Regression

In a supervise machine learning setting, the training dataDconsists ofNtraining instances each rep-
resented by a feature vectorx∈RMwithMfeatures and a labely∈R. Stochastic Gradient Decent based algorithms are most widely used to
optimize the object function (1).
## 0.3 Parallel Stochastic Gradient Descent

The PSGD learning algorithm is a very simple algorithm that learns in a distributed setting. Lets assume
we haveP workers. We assign one of the worker a role of master and others as workers. The role of
the master worker is to distribute the work i.e. Training data, and averages the local model learned by
each individual worker into a global model. Each worker on the other hand, gets its work from master
and learns a local model using SGD updates. Once an epoch (i.e. a complete round over the data) is
complete it sends back the local model to the master worker. The master after receiving models from all
the workers, averages them and sends the new global model to each worker for the second round. 

# 1 Parallel Linear Regression

The first task in this exercise is to implement a linear regression model and learn it using PSGD learning
algorithm explained above. You will implement it using mpi4py. A basic version of PSGD could be
though of using one worker as a Master, whose sole responsibility is getting local models from other
workers and averaging the model. A slight modification could be though of using all the workers as
worker and no separate master worker. [Hint: Think of collective routines that can help you in averaging
the models and return the result on each worker]. Once you implement your model, show that your
implementation can work for any number of workers i.e.P={ 2 , 4 , 6 , 8 }.

# 2 Performance and convergence of PSGD

The second task is to do some performance analysis and convergence tests.

1. First, you have to check the convergence behavior of your model learned through PSGD and
    compare it to a sequential version. You will plot the convergence curve (Train/Test score verses
    the number of epochs) forP ={ 1 , 2 , 4 , 6 , 7 }. You have to use any sequential version of Linear
    Regression forP= 1 (Only for sequential version you can use sklearn).
2. Second, you have to do a performance analysis by plotting learning curve (Train/Test scores) verses
    time. Time your program forP={ 1 , 2 , 4 , 6 , 7 }.
