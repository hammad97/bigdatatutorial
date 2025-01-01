
## Exercise Sheet 4
# Exercise 2: Analysis of Airport efficiency with Map Reduce 

In this exercise you will download data from Bureau of Transportation Statistics’ homepage^1. On the
Bureau’ homepage you will download data by selecting following fields:

- Filter geography: all
- Filter year: 2017

(^1) https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr


- Filter period: January
- Time period: FlightDate
- Airline: Reporting Airline, Flight Number Reporting Airline
- Origin: Origin
- Destination: Dest
- Departure Performance: DepTime, DepDelay
- Arrival Performance: ArrTime, ArrDelay

You will get a CSV file containing 450017 lines. An example of a line is as follows
1/1/2017 12:00:00 AM,AA,307,DEN,PHX,1135,-10,1328,-

In this exercise, you are going to write a python code usingHadoop MapReduceto accomplish several
requirements:

1. Computing the maximum, minimum, and average departure delay for each airport.[Hint: you are
    required to find max, min and avg in a single map reduce job]
2. Computing a ranking list that contains top 10 airports by their average Arrival delay.
3. What are yourmapper.pyandreduce.pysolutions?
4. Describe step-by-step how you apply them and the outputs during this procedure.

# Exercise 3: Analysis of Movie dataset using Map and Reduce 

For this exercise you will use movielens10m dataset available athttp://files.grouplens.org/datasets/
movielens/ml-10m-README.html. The movielens10m dataset consists of 10 million rating entries by
users. There are two main files required to solve this exercise 1) rating.dat and 2) movie.dat. The
rating.dat file contains userId, movieId and ratings (on scale of 1 to 5). The movie.dat file contains
information about movies i.e. movieId, Title and Genres.
In this exercise, you are going to write a python code usingHadoop MapReduceto accomplish several
requirements, note as your dataset is large you also need to perform some performance analysis i.e. how
many mappers and reducers you used, what is the performance increase by varying number of mappers
and reducers. Please also note that although you may have very few physical cores i.e. 2 or 4 or 8, but
you can still experiment with a large number of mappers and reducers.Plot the execution time vs
the number of mappers×reducers for each task below.

1. Find the movie title which has the maximum average rating?
2. Find the user who has assign lowest average rating among all the users who rated more than 40
    times?
3. Find the highest average rated movie genre? [Hint: you may need to merge/combine movie.dat
    and rating.dat in a preprocessing step.]


