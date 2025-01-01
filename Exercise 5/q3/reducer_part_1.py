#!/usr/bin/python3
import sys

total_r, num_r, max_avg = 0, 0, 0
max_id, m_id = None, None

# Following for loop is doing the main job which traverse through all data received
# from mapper task and keeps track of movie with their rating and in the end returns 
# the movie id with its max avg rating
for roww in sys.stdin:
    m, r = roww.split("\t")
    m = int(m)
    r = float(r)
    if m_id == m:
        num_r = num_r + 1
        total_r = total_r + r
    else:
        if m_id is not None:
            avgg  = total_r / num_r
            if avgg > max_avg:
                max_avg = avgg
                max_id = m_id
        num_r = 1
        total_r = r
        m_id = m

avgg  = total_r/num_r
if max_avg < (total_r / num_r):
    max_avg= total_r / num_r
    max_id = m_id

print('Maximum Average Rating : ')
print('Movie Id : ', max_id, ', Rating : ', max_avg)
