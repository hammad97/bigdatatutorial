#!/usr/bin/python3
import sys

total_r, num_r, max_avg = 0, 0, 0
g_id, max_id = None, None

# In this for loop like our other reducer job we are working on the output of mapper file
# Here we are aggregating ratings with respect to each genre and then calculating the average
# in the end whichever genre has highest rating we print the result in hadoop output directory. 
for roww in sys.stdin:
    g, r = roww.split("\t")
    r = float(r)
    if g_id == g:
        num_r = num_r + 1
        total_r = total_r + r
    else:
        if g_id is not None:
            c_avgg  = total_r / num_r
            if c_avgg > max_avg:
                max_avg = c_avgg
                max_id = g_id
        num_r = 1
        total_r = r
        g_id = g

if (total_r / num_r) > max_avg:
    max_avg = total_r / num_r
    max_id = g_id

print('Highest Average rated Movie Genre : ')
print('Genre : ', max_id, ', Rating : ', max_avg)