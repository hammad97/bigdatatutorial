import numpy as np 
from mpi4py import MPI
import os
import math
import csv

def load_20newsgroup(fPaths):
    docList = []
    for fPath in fPaths:
        for doc in os.listdir("20_newsgroups" + "/" + fPath):
            with open("20_newsgroups" + "/" + fPath + "/" + doc, 'rb') as f: 
                docList.append(f.read().decode("latin"))
    return docList

def doc_cleaner(docList):
    cleanData = []

    charac_with_space = '/,:;@.-?()!|"'
    charac_without_space = '#$%&\'*+<=>[\\]^_`{}~'

    for i in range(len(docList)):
        article = docList[i]

        for j in charac_with_space:
            article = article.replace(j, ' ')
            
        for j in charac_without_space:
            article = article.replace(j, '')
            
        
        article = article.lower()
        article = article.replace('\n', ' ')
        article = article.replace('\t', '')
        
        article = " ".join([word for word in article.split() if word not in common_eng_wrds])
        
        article = " ".join([word for word in article.split() if ((len(word)>1 and (not word.isdigit())))])
        cleanData.append(article)
    
    return cleanData

def doc_tokenizer(cleaned_documents):
    tokenizeData = []
    for doc in cleaned_documents:
        tokenizeData.append(doc.split())
    return tokenizeData

def export_CSV(tokenized_data, rank = 0):
    try:
        os.stat('tokenized_words')
    except:
        os.mkdir('tokenized_words')
    finally:
        with open('tokenized_words' + f'/rank_{rank}.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(tokenized_data)
        return None

def calc_tf(tokenized_data):
    tfVals = []
    for tok in tokenized_data:
        str_map = dict()
        for word in tok:
            str_map[word] = str_map.get(word, 0) + 1
        
        for word in str_map:
            str_map[word] = str_map[word] / len(tok)
        tfVals.append(str_map)
    return tfVals

def calc_df(tfVals):
    dfVals = {}
	
    for termF in tfVals:
        for term in termF:
            dfVals[term] = dfVals.get(term, 0) + 1
    return dfVals, len(tfVals)

def calc_idf(dfMap, totalSize):
    idfVals = {}
    for dfVal in dfMap: 
        idfVals[dfVal] = math.log(totalSize / dfMap[dfVal])
    return idfVals

def calc_tfidf(tfidfVals):
    def fetch_tfidf(word, docu_num):
        return tfidfVals[docu_num].get(word, 0)
    return fetch_tfidf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
exec_s_time = MPI.Wtime()

## words from common-english-words.txt
common_eng_wrds = ({'a','able','about','across','after','all','almost','also','am','among','an',
'and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did',
'do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her',
'hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like',
'likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on',
'only','or','other','our','own','rather','said','say','says','she','should','since','so','some',
'than','that','the','their','them','then','there','these','they','this','tis','to','too','twas',
'us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with',
'would','yet','you','your'})


timeMap = {}										# to store execution time for each operation

# 1.

s_time = MPI.Wtime()
if rank == 0:										# for master worker to split and send data to all slave workers
    newsList = os.listdir("20_newsgroups")

    newsList = np.array_split(np.array(newsList), size)
    for wrker in range(1, size):
        comm.send(newsList[wrker], dest = wrker, tag = 1)
    newsList = newsList[0]							# data master worker will be working on.
else:
    newsList = comm.recv(source = 0, tag = 1)		# data received by slave workers

timeMap['data_split'] = MPI.Wtime() - s_time		

w_name = ""
if rank == 0:
    w_name = "Master worker"
else:
    w_name = "Worker " + str(rank)
    
print('\nDataset assigned to ', w_name, ' : ', newsList)

s_time = MPI.Wtime()
read_data = load_20newsgroup(newsList)				# read data from each folder
timeMap['reading_time'] = MPI.Wtime() - s_time

s_time = MPI.Wtime()
cleaned_data = doc_cleaner(read_data)				# remove punctuations , stop words and digits from data
timeMap['cleaning_time'] = MPI.Wtime() - s_time

s_time = MPI.Wtime()
tokenized_data = doc_tokenizer(cleaned_data)		# tokenize the data
timeMap['tokenizing_time'] = MPI.Wtime() - s_time

s_time = MPI.Wtime()
export_CSV(tokenized_data, rank = rank)				# output tokenized words in csv
timeMap['save_csv_time'] = MPI.Wtime() - s_time

# 2.

s_time = MPI.Wtime()
term_freq = calc_tf(tokenized_data)					# calculate term frequency of tokenized data
timeMap['tf_calu_time'] = MPI.Wtime() - s_time

# 3.

s_time = MPI.Wtime()
dfVals, totalSize = calc_df(term_freq)				# calculate document frequency for each term
timeMap['df_calc_time'] = MPI.Wtime() - s_time

s_time = MPI.Wtime()
totalSize = comm.gather(totalSize, root = 0)		# gathering total size
df_gathered = comm.gather(dfVals, root = 0)			# gathering dfs

if rank==0:
    totalSize = sum(totalSize)	
    dfCollected = df_gathered[0]
    for dfVals in df_gathered[1:]:					# master worker combining all the input in one place
        for term in dfVals:
            dfCollected[term] = dfCollected.get(term, 0) + dfVals[term]

    idf = calc_idf(dfCollected, totalSize)			# calculation of idf with compiled data
else:
    idf = None

idf = comm.bcast(idf, root = 0)						# broadcasting idf val to workers

timeMap['idf_calculating_time'] = MPI.Wtime() - s_time

# 4.

s_time = MPI.Wtime()

tfidfList = []
for termF in term_freq:								# calculation of tf-idf 
    docu_tf_idf = {}
    for word in termF:
        docu_tf_idf[word] = termF[word] * idf[word]
    tfidfList.append(docu_tf_idf)
    
timeMap['tf_idf_cal_time'] = MPI.Wtime() - s_time

tfidf_gathered = comm.gather(tfidfList, root = 0)

if rank == 0:
    tfidfList = []
    for i in tfidf_gathered:
        tfidfList = tfidfList + i

timeMap['overall_time']  = MPI.Wtime() - exec_s_time
if rank == 0:
    print('\nMaster worker : ')
else:
    print('\nWorker ', rank, ' : ')

for k, v in timeMap.items():
    print(k , ' : ', v)

timeMaps = comm.gather(timeMap, root = 0)

if rank==0:    
    timeMap_e = {}
    for ind, i in enumerate(timeMaps):
        timeMap_e[ind] = i
        
    print(tfidfList[100]['logical'])  
    # for verify
    final_tf_idf = calc_tfidf(tfidfList)
    print('\ntfidf for \'logical\' : ', final_tf_idf('logical', 100))
    print('tfidf for \'a_dummy_term_which_should_return_0\' : ', final_tf_idf('a_dummy_term_which_should_return_0', 100))