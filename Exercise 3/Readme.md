
## Exercise Sheet 3
# Complex Data Lab: K-means clustering in a Distributed Setting

In this exercise sheet, you are going to apply distributed computing concepts using Message Passing
Interface API provided by Python (mpi4py) to Natural Language Processing (NLP) techniques using
complex data in the form of text documents. For this exercise sheet you willagain use “Twenty News-
groups” corpus(data set) available at UCI repositoryhttps://archive.ics.uci.edu/ml/datasets/
Twenty+Newsgroups.
In the previous lab you have done some pre-processing on the text data and converted the text data
into a tf-idf data. i.e. a vector representing a training example with tf-idf scores of each document.
Nowthe next task is to cluster these documents into groups. You will implement distributed
K-means clustering using MPI framework.
Note: If you are not sure about your preprocessing step in the last exercise you can use scikit-learn
[http://scikit-learn.org/stable/datasets/twenty_newsgroups.htmlONLYto](http://scikit-learn.org/stable/datasets/twenty_newsgroups.htmlONLYto) convert the text
into vector. You still have to implement K-means yourself.

# Distributed K-means Clustering 

The k-means algorithm clusters the data instances intokclusters by using euclidean distance between
data instances and the centroids of the clusters. The detail description of the algorithm is listed on slides
1-10https://www.ismll.uni-hildesheim.de/lehre/bd-16s/exercises/bd-02-lec.pdf. However,
in this exercise sheet you will implement a distributed version of the K-means. Figure below explains a
strategy to implement a distributed K-means.
Suppose you are given a DatasetX∈RM×Nand a random initial centroidsC∈RM×K, whereMare
the number of features of a Data instance,Nare the number of Data instances andKthe number of
clusters (K is a hyperparameter). Lets assume you want to implement a distributed version with 3
workers. (Noteyour solution should be generic and should work with any number of workers.) In the
figure below three workers are given colors i.e. Rank 0 = orange, Rank 1 = blue, and Rank 2 = green.
If a data is represented in white color this means it must be available on all the workers. The algorithm
progress as

- InitializeKcentroids


- Divide Data instances amongP workers (X shown in figure, different colors represent parts at
    different workers)
- Unitill converge
    - step 1
       ∗calculate distance of each Data instance from each centroid using the euclidean distance.
          (populate Distrance matrix shown in the figure)
       ∗Assign membership of each data instance using the minimum distance in the distance
          matrix.
    - step 2
       ∗Each worker calculates the new centroids (local means) using the new membership of data
          instance.
       ∗collect updated centroids (local mean) information from each worker and find the global
          centroids (global mean).
       ∗redistribute new centroids of clusters to each worker.


## Implement K-means

You have to implement distributed K-means clustering using MPI framework. Your solution should be
generic and should be able to run for arbitrary number of clusters. It should run in parallel i.e. not just
two workers working in parallel but all should participate in the actual work.

## Performance Analysis 

You have to do a performance analysis and plot a speedup graph. First you will run your experiments
with varying number of clusters i.e.P={ 1 , 2 , 4 , 6 , 8 }.


