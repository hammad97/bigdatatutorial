## Exercise Sheet 8

# Distributed Computing with Apache Spark

Apache Spark provides an abstraction called resilient distributed dataset (RDD). It provides two sets of
function to manipulate RDDs 1) Transformations and 2) Actions Apache Spark tutorial guide is available
athttps://spark.apache.org/docs/latest/rdd-programming-guide.html.

# Exercise 1: Apache Spark Basics 

## Part a) Basic Operations on Resilient Distributed Dataset (RDD) 

Let’s have two lists of words as follows:

- a = ["spark", "rdd", "python", "context", "create", "class"]
- b = ["operation", "apache", "scala", "lambda","parallel","partition"]

Create two RDD objects ofa,band do the following tasks. Words should be remained in the results
of join operations.

1. PerformrightOuterJoinandfullOuterJoinoperations betweenaandb. Briefly explain your
    solution. (1 point)
2. Usingmapandreducefunctions to count how many times the character"s"appears in allaand
    b. (1 point)
3. Usingaggregatefunction to count how many times the character"s"appears in allaandb. (
    point)

## Part b) Basic Operations on DataFrames 

Use datasetstudents.json(download from learnweb) for this exercise. First creating DataFrames from
the dataset and do several tasks as follows:

1. Replace thenullvalue(s) in columnpointsby the mean of all points.
2. Replace thenullvalue(s) in columndoband columnlastnameby"unknown"and"--"respectively. 
3. In the dob column, there exist several formats of dates, e.g.October 14, 1983and26 December 1989. Let’s convert all the dates into DD-MM-YYYY format where DD,MM and YYYY are two digits for day, two digits for months and four digits for year respectively. 
4. Insert a new columnageand calculate the current age of all students. 
5. Let’s consider granting some points for good performed students in the class. For each student, if his point is larger than 1 standard deviation of all points, then we update his current point to 20, which is the maximum. See Annex 1 for a tutorial on how to calculate standard deviation. 
6. Create a histogram on the new points created in the task 5. 

# Exercise 2: Manipulating Recommender Dataset with Apache Spark 

For this exercise you will use movielens10m dataset available athttps://grouplens.org/datasets/
movielens/10m/. The movielens dataset is a rating prediction dataset with ratings given on a scale of
1 to 5. Specifically, you will be working with Tags Data File Structuretags.dat, which contains data
in the form “UserID::MovieID::Tag::Timestamp”. You have to solve following questions using Apache
Spark transformations and actions.

1. A tagging session for a user can be defined as the duration in which he/she generated tagging
    activities. Typically, an inactive duration of 30 mins is considered as a termination of the tagging
    session. Your task is to separate out tagging sessions for each user.
2. Once you have all the tagging sessions for each user, calculate the frequency of tagging for each
    user session.
3. Find a mean and standard deviation of the tagging frequency of each user.
4. Find a mean and standard deviation of the tagging frequency for across users.
5. Provide the list of users with a mean tagging frequency within the two standard deviation from
    the mean frequency of all users.

