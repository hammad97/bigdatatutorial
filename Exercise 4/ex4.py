from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics as metr
import glob
import sys

# =============================================================================
# This is a generic function responsible for loading any of the two dataset
# and return it in a dataframe
# =============================================================================
def load_dataset(dt_selection):
    if dt_selection == 1:
        dtList = []
        virusData = glob.glob('dataset' + "/*.txt")
        for dt in virusData:
            dtList.append(pd.read_csv(dt, header = None))
        l_data = pd.concat(dtList, axis = 0, ignore_index = True)
        return l_data
    elif dt_selection == 2:
        l_data = pd.read_csv("cup98LRN.txt", header = None, low_memory = False)
        l_data = l_data.fillna(0)
        return l_data

# =============================================================================
# This function is specifically made for cleaning up the virus dataset, it also
# splits the training part which in our case is 70%. 
# =============================================================================
def clean_dataset1(l_data):
    rw, cl = np.shape(l_data)
    matX = np.zeros((rw, 500), dtype = float)
    matY = np.zeros((rw, 1))
    for i in range(0, rw):
        tmp = l_data.loc[i, 0].split()
        matY[i] = float(tmp[0])
        for j in range(1, np.size(tmp)):
            matX[i, int(tmp[j].split(':')[0])] = int(tmp[j].split(':')[1])
    xTrain = matX[0:int(rw * 0.7), :]
    yTrain = matY[0:int(rw * 0.7)]
    return matX, matY, xTrain, yTrain, int(rw * 0.7), rw
        
# =============================================================================
# This function handles the cleaning for kdd dataset as this dataset has alot of 
# columns filled with data which were not of great importance but it was making
# the code take longer for processing so i decided to remove that data
# =============================================================================
def clean_dataset2(l_data):
    l_data = np.delete(l_data, np.s_[0:16], 1)
    l_data = np.delete(l_data, np.s_[1:7], 1)
    l_data = np.delete(l_data, 3, 1)
    l_data = np.delete(l_data, np.s_[27:30], 1)
    l_data = np.delete(l_data, np.s_[29:49], 1)
    l_data = np.delete(l_data, np.s_[315:388], 1)
    l_data = np.delete(l_data, 350, 1)
    l_data = np.delete(l_data, np.s_[352:361], 1)
    l_data = np.delete(l_data, 0, 0)
    l_data[l_data == ' '] = '0'
    rw, cl = np.shape(l_data)
    matX = np.zeros((rw, 351), dtype = float)
    matY = np.zeros((rw, 1))
    for i in range(0, cl):
        l_data[:, i] = l_data[:, i].astype(float)
        if i == 351:
            matY[:, 0] = (l_data[:, i])
        else:
            matX[:, i] = (l_data[:, i])
    return matX, matY, int((rw) * 0.7), rw
        
# =============================================================================
# This function is responsible for showing the final output which contains the plot too
# =============================================================================
def show_result(num_iter, tme, trainLoss, testLoss):
	print("Total Iterations : ", num_iter)
	print("Training RMSE : ", trainLoss[num_iter - 1])
	print("Testing RMSE : ", testLoss[num_iter - 1])
	print("Total time : ", tme)
	
	plt.plot(trainLoss, 'b', label = 'RMSE Training') 	
	plt.plot(testLoss, 'g', label = 'RMSE Testing')
	plt.title('RMSE Training and Testing')
	plt.xlabel('Number of Epochs')
	plt.ylabel('RMSE')
	plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
	plt.show()	
	return None

# =============================================================================
# This is the main function of our program here we are initializing MPI variables,
# wuth other variables which will be used through PSGD implementation
# =============================================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
max_iter = 50
trainLoss = [] 
testLoss = [] 
isConverge = False

dt_selection = int(sys.argv[1]) #dataset cmd line param  1 : virus, 2 : kdd

# We broadcase dataset command line param to other workers as well
dt_selection = comm.bcast(dt_selection, root = 0)

# =============================================================================
# Here we are calling the load and clean function for virus dataset and preparing
# the 70% train data with weights and learning rate
# =============================================================================
if dt_selection == 1:
    if rank == 0:
        l_data = load_dataset(dt_selection)
        matX, matY, xTrain, yTrain, dt_70, rw = clean_dataset1(l_data)
        merged_data = np.concatenate((yTrain, xTrain), axis = 1)
        matX_70 = matX[dt_70:rw, :]
        matY_70 = matY[dt_70:rw]
    i_weights = np.zeros((1, 500), dtype = float)
    lr = 10**-13
	
# =============================================================================
# Here we are loading and cleaning the kdd dataset by using our functions and
# preparing the train data with initial weights and learning rate
# =============================================================================
if dt_selection == 2:
    if rank == 0:
        l_data = load_dataset(dt_selection)
        l_data = l_data.values
        matX, matY, dt_70, rw = clean_dataset2(l_data)
        xTrain = matX[0:dt_70, :]
        yTrain = matY[0:dt_70]
        merged_data = np.concatenate((yTrain, xTrain), axis = 1)
        matX_70 = matX[dt_70:rw, :]
        matY_70 = matY[dt_70:rw]
    i_weights = np.zeros((1, 351), dtype=float)
    lr = 10**-16

# Start measuring time at this point when SGD starts
if rank == 0:
	s_time = MPI.Wtime()
		
# =============================================================================
# This while loop is the main part which is responsible for implementation of SGD
# Here we firstly splitting up the data and sending that with weights to all slave 
# workers using collective communication. Then we receive each dataset on every worker
# and after some shuffling we use the forward pass and backpropagation as mentioned in algo
# to compute the updated weights then master worker receives all updated weights with
# predicted data. Now master node checks the criteria for convergance (i.e if max iter reached
# or loss is less than 10e-7) if convergence criteria is matched we update our convergence flag
# and broadcast updated value of this flag with all workers so all the workers halt.
# Then use master worker to invoke our plotting function
# =============================================================================
j = 0
while (not isConverge) and (j < max_iter):
	if rank == 0:
		splt = np.array_split(merged_data, size)
	else:
		splt = None
	wgts_r = comm.bcast(i_weights, root = 0)
	splt_r = comm.scatter(splt, root = 0)
	np.take(splt_r, np.random.permutation(splt_r.shape[0]), axis = 0, out = splt_r)
	srw, scl = np.shape(splt_r)
	pred = np.zeros((srw, 1))
	for i in range(0, srw):
		xTrain = splt_r[i, 1:501]
		yTrain = splt_r[i, 0]
		pred[i, 0] = np.matmul(xTrain, wgts_r.T)
		wgts_r = wgts_r - lr * ((-2 * (yTrain - pred[i, 0])) * xTrain)
	
	c_wghts = comm.gather(wgts_r, root = 0)
	predY = comm.gather(pred, root = 0)
	tVal = comm.gather(splt_r[:, 0], root = 0)

	if rank == 0 and isConverge == False:
		i_weights = np.mean(c_wghts, axis = 0)
		trLoss = np.sqrt(metr.mean_squared_error(np.hstack(tVal), np.vstack(predY)))
		trainLoss.append(trLoss)
		tsLoss = np.sqrt(metr.mean_squared_error(matY_70, np.matmul(matX_70, i_weights.T)))
		testLoss.append(tsLoss)	
		print("epoch # ", j, ", Loss : ", abs(testLoss[j - 1] - testLoss[j]))
		if j > 0 and abs((testLoss[j - 1] - testLoss[j])) < 10**-7:
			print('Converging with loss: ', testLoss[j - 1] - testLoss[j])
			isConverge = True
	isConverge = comm.bcast(isConverge,root=0)
	j += 1

if rank == 0:
	show_result(j, MPI.Wtime() - s_time, trainLoss, testLoss)




