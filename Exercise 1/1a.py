import numpy as np
from mpi4py import MPI

n = 4
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
v_u = np.random.random((10**n, 1))
v_v = np.random.random((10**n, 1))

if rank == 0:   #For master worker
    if size == 1:           #For master worker when there is only one worker awho has to do tasks alone.
        s_time = MPI.Wtime()
        v_merged = np.hstack((v_u, v_v))
        # print(v_merged)
        comm.send(v_merged, dest = 0)
        res = comm.recv(source = 0)
        comm.send(res[:, 0] + res[:, 1], dest = 0)
        v_z = comm.recv(source = 0)
        print("Time taken : {} when  P = {} and n = {}" .format(MPI.Wtime() - s_time, size, n))
    else:                   #For master worker when it divides all tasks among its slaves.
        s_time = MPI.Wtime()   
        v_merged = np.hstack((v_u, v_v))
        # print(v_merged)
        v_splt = np.array_split(v_merged, size - 1)
        for i in range(1, size):
            comm.send(v_splt[i - 1], dest = i)
elif rank != 0:             # For other than master workers
    if size > 1:
        res = comm.recv(source = 0)
        summ = res[:, 0] + res[:, 1]
        comm.send(summ, dest = 0)
	

if rank == 0 and size > 1:   # For master worker when it receives task from all slaves to compile result and show.
	v_z = comm.recv(source = 1)
	for i in range(2, size):
		sum_r = comm.recv(source = i)
		v_z = np.hstack((v_z, sum_r))
	# print(v_z)
	print("Time taken : {} when  P = {} and n = {}" .format(MPI.Wtime() - s_time, size, n))
