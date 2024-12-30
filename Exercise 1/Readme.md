
## Exercise Sheet 1

# Topic: Distributed Computing with Message Passing Interface

# (MPI)

In this exercise sheet, you will solve problems using Message Passing Interface API provided by Python
(mpi4py). The lecture slides provides the basic introduction to the APIs. In the Annex section there
are few useful resources that will help you further understand MPI concepts and provide help in solving
following exercises.


# Exercise 1: Basic Parallel Vector Operations with MPI 

Suppose you are given a vectorv∈RN. Initialize your vectorvwith random numbers (can be either inte-
gers or floating points). You will experiment with three different sizes of vector, i.e.N={ 107 , 1012 , 1015 }.
You have to parallelize vector operations mentioned below using MPI API. For each operation you will
have to run experiment with varying number of workers, i.e. if your system hasP workers than run
experiments with workers ={ 1 , 2 ,... P}for each size of vector given above. You have to time your
code for each operation and present it in a table. [Note: You have to define/explain your parallelization
strategy i.e. how you assign task to each worker, how you divide your data etc.]. You have to use MPI
point-to-point communication i.e. Send and Recv.

a) Add two vectors and store results in a third vector.


b) Find an average of numbers in a vector.

Note: Your code should solve a correct problem. To be sure you are doing everything correct, i.e.
choosen= 16 to verify if your results are correct. Incorrect solutions will not earn you any points.

# Exercise 2: Parallel Matrix Vector multiplication using MPI 

In this exercise you have to work with a matrixA∈RN×Nand two vectorsb∈RN,c∈RN. Initialize
the matrixAand vectorbwith random numbers (can be either integers or floating points). The vector
cwill store result ofA×b.
In case of matrix vector multiplication, you will experiment with three different sizes of matrices i.e.
N={ 102 , 103 , 104 }. [note: your matrix will beN×N, which means in case 1 you will have matrix
dimension 100x100]. You will have to run experiments with varying number of workers, i.e. if your system
hasPworkers than run experiments with workers ={ 1 , 2 ,... P}for each matrix size given above. You
have to time your code and present it in a table.
Implement parallel matrix vector multiplication using MPI point-to-point communication i.e. Send
and Recv. Explain your logic in the report i.e. how the matrix and vectors are divided (distributed)
among workers, what is shared among them, how is the work distributed, what individual worker will
do and what master worker will do.

Note: depending on your system RAM you might not be able to run experiment with matrix size
n= 10^4.

# Exercise 2: Parallel Matrix Operation using MPI 

In this exercise you have to work with three matrices (A∈RN×N,B∈RN×N,C∈RN×N) i.e each
matrix having sizeN×N. Initialize your matricesAandBwith random numbers (can be either integers
or floating points). MatrixCwill store result ofA×B.
In case of matrix multiplication, you will experiment with three different sizes of matrices i.e.N=
{ 102 , 103 , 104 }. [note: your matrix will beN×N, which means in case 1 you will have matrices with
dimension 100x100]. You will have to run experiments with varying number of workers, i.e. if your system
hasPworkers than run experiments with workers ={ 1 , 2 ,... P}for each matrix size given above. You
have to time your code and present it in a table.
Implement parallel matrix matrix multiplication using MPI collective communication. Explain your
logic in the report i.e. how the matrices are divided (distributed) among workers, what is shared among
them, how is the work distributed, what individual worker will do and what master worker will do. Per-
form experiments with varying matrix sizes and varying number of workers. You can look at the imple-
mentation provided in the lecture (slide 48)https://www.ismll.uni-hildesheim.de/lehre/bd-17s/
script/bd-01-parallel-computing.pdf

Note: depending on your system RAM you might not be able to run experiment with matrix size
n= 10^4.

