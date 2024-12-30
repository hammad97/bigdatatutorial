import numpy as np
from mpi4py import MPI
import os
from sklearn.feature_extraction.text import TfidfVectorizer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
c_centroid = 5
max_iter = 45

# =============================================================================
# Reading all the dataset from folders as done in prev exercise
# and using TfidfVectorizer to generate tfidf values as instructed.
# =============================================================================
def load_20newsgroup():
    newsList = os.listdir("20_newsgroups")
    docList = []
    for fPath in newsList:
        for doc in os.listdir("20_newsgroups" + "/" + fPath):
            with open("20_newsgroups" + "/" + fPath + "/" + doc, 'rb') as f: 
                docList.append(f.read().decode("latin"))

    tfidf_vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 2,
                                        max_features = 10000,
                                        stop_words = 'english')

    tfidf = tfidf_vectorizer.fit_transform(docList)
    return tfidf

# =============================================================================
# Loading the data first by master worker using load_20newsgroup() and then 
# initializing k centroids using initialize_Kcentroids() then using master worker
# to split the data and pass that to slave workers
# =============================================================================
def data_load_split(c_centroid, comm = comm):
    i_time = MPI.Wtime()
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        newsgrp_data = load_20newsgroup()
    else:
        newsgrp_data = None

    load_time = MPI.Wtime() - i_time 
    centroidList, centroids_i_time = initialize_Kcentroids(newsgrp_data, c_centroid)
    
    s_time = MPI.Wtime()
    if rank == 0:
        splt_data = []
        splt_size = newsgrp_data.shape[0]//size
        for i in range(size):
            if i < size - 1:
                segmnt = slice(i * splt_size,(i + 1) * splt_size)
            else:
                segmnt = slice(i * splt_size, None)
            
            splt_data.append(newsgrp_data[segmnt])
        newsgrp_data = splt_data
    else:
        newsgrp_data = None

    newsgrp_data = comm.scatter(newsgrp_data, root = 0)
    splt_time = MPI.Wtime() - s_time 

    return newsgrp_data, centroidList, load_time, splt_time, centroids_i_time

# =============================================================================
# Here we are initiating centroids from master worker with respect to number of mentioned centroids
# then we are broadcasting these centroids to all slave workers
# =============================================================================
def initialize_Kcentroids(newsgrp_data, c_centroid, comm = comm):
    rank = comm.Get_rank()
    s_time = MPI.Wtime()
    if rank == 0:
        rand_data = np.random.choice(np.arange(newsgrp_data.shape[0]), size = c_centroid, replace = False)
        
        centroidList = []
        for i in rand_data:
            centroidList.append(newsgrp_data[i].A)
        centroidList = np.vstack(centroidList)
    else:
        centroidList = None

    centroidList = comm.bcast(centroidList, root = 0)
    centroids_i_time = MPI.Wtime() - s_time
    return centroidList, centroids_i_time

# =============================================================================
# This function is simply responsible for calculation of distance with data and the centroids
# =============================================================================
def calc_distance(newsgrp_data, centroidList):
    distanceRes = []
    for i in range(newsgrp_data.shape[0]):
        dlta = newsgrp_data[i].A - centroidList
        distanceRes.append(np.sqrt(np.sum(np.power(dlta, 2), axis = -1)))
    return np.array(distanceRes)

# =============================================================================
# Here we are calculating/updating the centroids based on their distances
# and checking if any of them are converging at the same time to stop before maximum
# iterations are reached. Other condition such as when there isnt any data belonging 
# to a centroid is handled. And lastly master worker is gathering and updating centroids to later
# broadcast updated centroids with all the slave workers. This happens until convergence or maximum
# iterations are reached (whichever comes first) 
# =============================================================================
def calc_centroids(newsgrp_data, centroidList, c_centroid, max_iter = max_iter, comm = comm):
    rank = comm.Get_rank()
    prev_member = np.ones(shape = (newsgrp_data.shape[0],))
    memberList = np.arange(c_centroid)[...,np.newaxis]

    start_loop = MPI.Wtime()
    for i in range(max_iter):
        distanceArr = np.array(calc_distance(newsgrp_data, centroidList))
        membr = np.argmin(distanceArr, axis = -1)
        membrMsk = (membr[np.newaxis,...])==(memberList)
        
        if np.all(membr == prev_member) & (i > 0):
            converged = True
        else:
            prev_member = membr
            converged = False

        converged_all = comm.allgather(converged)
        if np.all(np.vstack(converged_all)):
            print('Worker ', rank, ' converged in ', i, ' iterations')
            break
        
        membershipInstances = np.zeros(shape=(c_centroid,))

        for cntr in range(c_centroid): 
            membershipInstances[cntr] = np.sum(membrMsk[cntr])
            if  membershipInstances[cntr] > 0:
                centroidList[cntr] =  newsgrp_data[membrMsk[cntr]].mean(axis = 0)
            else:
                centroidList[cntr] = 0
    
        centroidList = comm.gather(centroidList, root = 0)
        membershipInstances = comm.gather(membershipInstances, root = 0)
        
        if rank == 0:
            centroidList = np.array(centroidList)
            membershipInstances = np.array(membershipInstances)
            print('Iteration : ', i, ' -> Membership Data: ', membershipInstances)
            membershipInstances = membershipInstances / np.sum(membershipInstances, axis = 0)
            centroidList = np.einsum('wcd,wc->cd', centroidList, membershipInstances)
        else:
            pass
            
        centroidList = comm.bcast(centroidList, root = 0)

    if not converged_all:
        distanceArr = np.array(calc_distance(newsgrp_data, centroidList))
    else:
        pass
    avg_time = (MPI.Wtime() - start_loop) / i

    return centroidList, distanceArr, avg_time, i

# =============================================================================
# Here we are calculating the distortion which is sum of squared distances
# our objective here is to minimize this value.
# =============================================================================
def calc_distortion(distanceArr, centroidList, comm = comm):
    s_time = MPI.Wtime()
    rank = comm.Get_rank()
    c_centroid = centroidList.shape[0]
    membr = np.argmin(distanceArr, axis = -1)
    distanceArr = np.min(distanceArr, axis = -1)
    memberList = np.arange(c_centroid)[...,np.newaxis]
    membrMsk = membr[np.newaxis,...] == memberList

    c_square = np.zeros(shape = (c_centroid,))
    for cntr in range(c_centroid):
        c_square[cntr] = np.sum(np.power(distanceArr[membrMsk[cntr]], 2))
    
    c_square = comm.gather(c_square, root = 0)
    if rank == 0:
        c_square = np.array(c_square)
        c_square = np.sum(c_square)
    else:
        pass
    
    time_taken = MPI.Wtime() - s_time
    return c_square, time_taken



# Works for multiple number of centroids
if type(c_centroid) != list:        
    timeMap = {}
    newsgrp_data, centroidList, load_time, splt_time, centroids_i_time = data_load_split(c_centroid)
    timeMap['dataload'] =  load_time
    timeMap['datasplit'] =  splt_time
    timeMap['centroids_i_time'] = centroids_i_time

    centroidList, distanceArr, avg_time, iterations = calc_centroids(newsgrp_data, centroidList, c_centroid, max_iter)
    timeMap['time_per_iter'] = avg_time
    timeMap['iterations'] = iterations

    timeMap = comm.gather(timeMap, root = 0)

    if rank == 0:
        for i in range(len(timeMap)):
            print('Worker ', i , ' : ',timeMap[i])
else:           
    for cntr in c_centroid:
        timeMap = {}
        newsgrp_data, centroidList, load_time, splt_time, centroids_i_time = data_load_split(cntr)
        timeMap[f'dataload_{cntr}'] =  load_time
        timeMap[f'datasplit_{cntr}'] =  splt_time
        timeMap[f'centeroids_initailization_time_{cntr}'] = centroids_i_time

        centroidList, distanceArr, avg_time, iterations = calc_centroids(newsgrp_data, centroidList, cntr, max_iter)
        timeMap[f'time_per_iter_{cntr}'] = avg_time
        timeMap[f'iterations_{cntr}'] = iterations
        
        c_square, calc_distort_t = calc_distortion(distanceArr, centroidList)

        timeMap[f'distortion_calc_{cntr}'] = calc_distort_t
        timeMap[f'distortion_{cntr}'] = c_square

        timeMap = comm.gather(timeMap, root = 0)

        if rank == 0:
            for i in range(len(timeMap)):
                print('Worker ', i , ' : ',timeMap[i])