import numpy as np
from mpi4py import MPI

n = 4
dim = 16
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
v_c = np.zeros((dim, 1))
v_v = np.zeros(shape = (dim, 1))
matA = np.zeros(shape = (dim, dim))

# matA = np.arange(16*16).reshape(16, 16)
# v_v = np.arange(16).reshape(16, 1)


if rank == 0:                           # For master worker
    if size == 1:                       # Master worker performs all operation itself when only one worker
        s_time = MPI.Wtime()
        matA = np.random.random((dim, dim))
        v_v = np.random.random((dim, 1))
        # print("matA : ", matA)
        # print("vecV : ", v_v)
        comm.send(matA, dest = 0)
        res = comm.recv(source = 0)
        vector_s = np.zeros(shape=(dim, 1))
        for i in range(0, dim):       # multiplication implementation 
            for j in range(0, dim):
                vector_s[i, 0] += (res[i, j] * v_v[j, 0])
        comm.send(vector_s, dest = 0)
        v_c = comm.recv(source = 0)
        # print("vecC : ", v_c)	
        print("Time taken : {} when  P = {} and n = {}" .format(MPI.Wtime() - s_time, size, n))
    elif size > 1:                      # When more than one worker, then master handout tasks to slave workers
        s_time = MPI.Wtime()
        matA = np.random.random((dim, dim))
        # print("matA : ", matA)
        v_splt = np.array_split(matA, size - 1)
        for i in range(1, size):
            comm.send(v_splt[i - 1], dest = i)
elif rank != 0:                         # For slave workers to receive the task from master and send back the result
    if size > 1:
        v_v = np.random.random((dim, 1))
        # print("vecV : ", v_v)
        res_m = comm.recv(source = 0)
        res_mul = np.matmul(res_m, v_v)
        comm.send(res_mul, dest = 0)
        
if rank == 0 and size > 1:              # Master worker receives results from slave workers and compile + display them
    v_c = comm.recv(source = 1)
    for j in range(2, size):            # compiling pieces of results into final result
        result_s = comm.recv(source = j)
        v_c = np.vstack((v_c, result_s))
    # print("vecC : ", v_c)
    print("Time taken : {} when  P = {} and n = {}" .format(MPI.Wtime() - s_time, size, n))
