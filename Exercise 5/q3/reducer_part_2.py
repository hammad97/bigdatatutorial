#!/usr/bin/python3
import sys

min_avg, total_r = 5, 0         # taking the highest val for min_avg in start
min_id, u_id = None, None

# In this for loop the reducer is accepting data from mapper and on each line of this data
# the reducer extracts the userid with its rating and makes a counter of the ratings from that user
# once that counter gets more than 40 then we calculate the average and if that average is lower
# than previous stored average then we update our min_avg and in the end we print the result to hadoop output dir. 
for roww in sys.stdin:
    u, r = roww.split("\t")
    u = int(u)
    r = float(r)
    if u_id == u:
        r_counter = r_counter + 1
        total_r = total_r + r
    else:
        if u_id is not None:
            if r_counter > 40:
                c_avg  = total_r / r_counter
                if c_avg < min_avg:
                    min_avg = c_avg
                    min_id = u_id
        r_counter = 1
        total_r = r
        u_id = u

if r_counter > 40:
    if (total_r / r_counter) < min_avg:
        min_avg = (total_r / r_counter)
        min_id = u_id

print('Lowest Average Rating : ')
print('User id : ', min_id, ', Rating : ', min_avg)