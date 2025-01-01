#!/usr/bin/python3
import sys

# Here i am simply reading the ratings.dat file again and printing the output
# which is being used in our reducer to find max average rating of the movie.
# unlike previous mapper task here i am outputting user with rating.
for roww in sys.stdin:
    roww = roww.strip()
    u, m, r, _ = roww.split("::")
    u = int(u)
    r = float(r)
    print(f'{u}\t{r}')
