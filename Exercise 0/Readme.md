
## Exercise Sheet 0

# Exercise 1: Pandas and Numpy 

- Matrix Multiplication: Create a numpy matrix A of dimensionsn×m, wheren= 100 and
    m= 20. Initialize Matrix A with random values. Create a numpy vector v of dimensionsm×1.
    Initialize the vector v with values from a normal distribution usingμ= 2 andσ= 0.01. Perform
    the following operations:
       1. Iteratively multiply (element-wise) each row of the matrix A with vector v and sum the result
          of each iteration in another vector c. This operation needs to be done with for-loops, not
          numpy built-in operations.
       2. Find mean and standard deviation of the new vector c.
       3. Plot the histogram of vector c using 5 bins.
- Grading Program: This task puts you in the position that I end up at the end of every semester.
    Which is, grading your work and issuing the grades. In this task you are required to use the
    ‘Grades.csv’ File that has been provided on learnweb.
- Read the data from the csv.
- For each student,
    - Compute the sum for all subjects for each student.
    - Compute the average of the point for each student. (total points are 500).
    - Compute the standard deviation of point for each student.
    - Plot the average points for all the students (in one figure).
    - For each student assign a grade based on the following rubric.
    - Plot the histogram of the final grades.

# Exercise 2: Linear Regression through exact form. 

In this exercise, you will implement linear regression that was introduced in the introduction Machine
Learning Lecture. The method we are implementing here today is for a very basic univariate linear
regression.

- Generate 3 sets of simple data, each consisting of a matrix A with dimensions 100×2. Initialize
    the sets of data with normal distributionμ= 2 andσ= [0. 01 , 0. 1 ,1] so that each dataset has a
    differentσ. You may assume that the first column of A represents the predictor data (X), whereas
    the second column of matrix A represents the target data (Y).
- Implement LEARN-SIMPLE-LINREG algorithm and train it using matrix A to learn values ofβ 0
    andβ 1
- Implement PREDICT-SIMPLE-LINREG and calculate the points for each training example in
    matrix A.
- Plot the training data (use plt.scatter) and your predicted line (use plt.plot).
- Putβ 0 to zero and rerun the program to generate the predicted line. Comment on the change you
    see for the varying values ofσ
- Putβ 1 to zero and rerun the program to generate the predicted line. Comment on the change you
    see for the varying values ofσ
- Use numpy.linalg lstsq to replace step 2 for learning values ofβ 0 andβ 1. Explain the difference
    between your values and the values from numpy.linalg lstsq.



