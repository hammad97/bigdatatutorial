import numpy as np
from mpi4py import MPI

n = 4
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
v_u = np.random.random((10**n, 1))
v_v = np.random.random((10**n, 1))

if rank == 0:               # Condition for master worker
    if size == 1:           # When there is only one worker. Doing all the task itself.
        s_time = MPI.Wtime()
        # print(v_v)
        comm.send(v_v, dest = 0)
        v_res = comm.recv(source = 0)
        comm.send(np.sum(v_res), dest = 0)
        comm.send(len(v_res), dest = 0)
        vv_sum = comm.recv(source = 0)
        ss_sum = comm.recv(source = 0)
        # print(vv_sum / ss_sum)
        print("Time taken : {} when  P = {} and n = {}" .format(MPI.Wtime() - s_time, size, n))
    else:                   # When more than one worker then dividing the task among the slaves.
        s_time = MPI.Wtime()
        # print(v_v)
        if size > 1:
            v_splt = np.array_split(v_v, size - 1)
        for i in range(1, size):
            comm.send(v_splt[i - 1], dest = i)
elif rank != 0:             # Condition for slave workers
    if size > 1:            # To recieve the matrix and perform required operation and return the result back to master
        v_res = comm.recv(source = 0)
        comm.send(np.sum(v_res), dest = 0)
        comm.send(len(v_res), dest = 0)        
	
if rank == 0 and size > 1:  # For master worker to receive all results from slave workers and compile + display result.
    vv_sum = comm.recv(source = 1)
    ss_sum = comm.recv(source = 1)
    for i in range(2, size):    
        sum_s = comm.recv(source = i)
        len_s = comm.recv(source = i)
        vv_sum = np.hstack((vv_sum, sum_s))
        ss_sum = np.hstack((ss_sum, len_s))
    print("Time taken : {} when  P = {} and n = {}" .format(MPI.Wtime() - s_time, size, n))
    # print(np.sum(vv_sum) / np.sum(ss_sum))