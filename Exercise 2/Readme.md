## Exercise Sheet 2

In this exercise sheet, you are going to apply distributed computing concepts using Message Passing
Interface API provided by Python (mpi4py) to Natural Language Processing (NLP) techniques using
complex data in the form of text documents. The NLP application uses machine learning models to
understand the human language and speech. The text data is usually large and consists of complex
structures. In this lab you will use MPI framework to process large natural language corpora.
More precisely, you are going to do some basic tasks in NLP including data cleaning, text tokenization
and convert words into their Term Frequency, Inverse Document Frequency (TFIDF) scores.

# Dataset and a scikit-learn

You will use “Twenty Newsgroups” corpus (data set) available at UCI repositoryhttps://archive.
ics.uci.edu/ml/datasets/Twenty+Newsgroups. It consists of 20 subfolders each belong to a specify
topic. Each subfolder has multiple documents.

You can look at the the following blog post to get yourself familiarized with TFIDF calculation avail-
able in scikit-learn (https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a).
Note: This is just a tutorial you will not use scikit-learn to solve the final task.

# Exercise 1: Data cleaning and text tokenization 

In a text document you encounter words that are not helpful for your final model. For example, punctu-
ations and numbers, meaningless words, common English stopwords etc. Your first task is to preprocess
your data so you can remove these words from the documents. Your solution should be based on MPI
framework. You have to develop (write code) a distributed algorithm that cleans the data by removing


the above mentioned words/numbers. You can take help of some python libraries i.e. NLTK. The devel-
oped program should take a set of documents in raw format as input, and outputs a set of documents
that are cleaned and tokenized.
Please explain your solution and how you distribute work among workers.

1. Cleaning: remove all punctuations, numbers and The list of common English stopwords used in
    this exercise sheet can be found in the reference [4].
2. Tokenize: Tokenize your documents so it is easy to process in the next task i.e. Tokenize words
    and output as a comma separated document.

# Exercise 2: Calculate Term Frequency (TF) 

The Term Frequency (TF) is calculated by counting the number of times a token occurs in the document.
This TF score is relative to a specific document, therefore you need to normalized it by dividing with
the total number of tokens appearing in the document. A normalized TF score for a specific tokentin
a documentdcan be calculated as,

```
TF(t, d) =
```
```
nd(t)
∑
t′∈dn
d(t′), (1)
```
wherend(t) is the number of times a tokentappears in a documentdand|d|is the total number of
tokens in the documentd.
Develop an solution using MPI framework and write code. Please explain how you parallelize (or dis-
tribute) TF(t,d) calculation. Also explain your strategy from the data division and calculation division
point of view as well. Perform a small experiment by varying number workers i.e. P={ 2 , 4 , 8 }. Also
verify if you get the same result at the end.

# Exercise 3: Calculate Inverse Document Frequency ((IDF) 

The Inverse Document Frequency ((IDF) is counting the number of documents in the corpus and counting
the number of documents that contain a token. While the TF is calculated on a per-document basis, the
IDF is computed on the basis of the entire corpus. The final IDF score of a tokentin the corpusCis
obtained by taking logarithm of document count in the corpus divided by the number of documents in
the corpus that contain a particular token. 

Develop an solution using MPI framework and write cod. Please explain how you parallelize (or
distribute) IDF(t,d) calculation. Also explain your strategy from the data division and calculation
division point of view as well. Perform a small experiment by varying number workers i.e.P={ 2 , 4 , 8 }.
Also verify if you get the same result at the end.

# Exercise 4: Calculate Term Frequency Inverse Document Frequency (TF-IDF) scores 

In this exercise you will find the TF-IDF(t,d) for a given tokentand a documentdin the corpusCis
the product of TF(t,d) and IDF(t), which is represented as,

```
TF-IDF(t, d) =TF(t, d)×IDF(t) , (3)
```
You have already calculatedTF(t, d) in Exercise 2 andIDF(t) is Exercise 3.
In this exercise you have to think about how you can combine the complete pipeline in a single paral-
lel/distributed program that can run workerP ={ 2 , 4 , 8 }. Develop an solution using MPI framework


and write code. Please explain how you parallelize (or distribute) the complete pipeline. Also explain
your strategy from the data division and calculation division point of view as well. Perform a small
experiment by varying number workers i.e. P={ 2 , 4 , 8 }. Also verify if you get the same result at the
end.

