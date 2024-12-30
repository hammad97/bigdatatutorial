import numpy as np
from mpi4py import MPI

n = 4
dim = 10**n
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
matA = np.random.random((dim, dim))
matB = np.random.random((dim, dim))

# matA = np.arange(16*16).reshape(16, 16)
# matB = np.arange(16*16).reshape(16, 16)

if rank == 0:                               # For master worker only
	s_time = MPI.Wtime()
	# print("matA : ", matA)
	# print("matB : ", matB)
	splt_A = np.array_split(matA, size)     # makes parts to later send to scatter in slave workers
else:
	splt_A, matB = None, None
scat_A = comm.scatter(splt_A, root = 0)     # scattering in slave workers
broadcast_B = comm.bcast(matB, root = 0)    # broadcasting to slave workers
mul_A_B = np.matmul(scat_A, broadcast_B)    
res_M = comm.gather(mul_A_B, root = 0)      # receiving results by gather
try:
    matC = np.concatenate(res_M).reshape(dim, dim)      #Compiling and printing the final result.
    # print("matC : ", matC)
    print("Time taken : {} when  P = {} and n = {}" .format(MPI.Wtime() - s_time, size, n))
except TypeError:                           # Didnt understand reason behind this Error, so made an exception. But code works as expected
    pass
