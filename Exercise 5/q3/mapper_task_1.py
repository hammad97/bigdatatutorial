#!/usr/bin/python3
import sys

# Here i am simply reading the ratings.dat file and printing the output
# which is being used in our reducer to find max average rating of the movie.
# Here i am outputing movie with its rating.
for roww in sys.stdin:
    roww = roww.strip()
    u, m, r, _ = roww.split("::")
    m = int(m)
    r = float(r)
    print(f'{m}\t{r}')
