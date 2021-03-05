# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:27:55 2021

@author: P.Akash Pattanaik
    
AI_coding_assignment_MNC: converting to json format
"""
# Import required libraries
import json # for importing data from json file
import os
import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import pdist, squareform, cosine
import itertools
from sklearn.preprocessing import MinMaxScaler
from nltk.stem.wordnet import WordNetLemmatizer # for finding the root of the words
from nltk.corpus import stopwords # to find stop words
from nltk.tokenize import word_tokenize
from string import punctuation 
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# read the data
path = r'C:/Users/User/Desktop/Akash Desktop/documents/Company/MNC group/' 
# file = open(path + "data.json", 'r')
#data = json.load(json_file)

##_______________________________________________________________________________________________________
## as the data is not in proper json format hence it is converted to json format
## this code has been removed as the new data is in json format
# lines = file.readlines()
# li = []

# for x in lines:
#     #print(x)
#     #y = json.load(x)
#     #json.dumps(x, outfile)
#     #li.append(x.strip())
    
# #print(li[2])
#     with open("dataJson.json", "a") as fout:
#         fout.write(x.strip() + ',')
##_______________________________________________________________________________________________________

## Read the new json data and csv file   
file = open(path + "json_data.json", 'r', encoding="utf8")
data_json = json.load(file)
user_profiles = pd.read_csv(path + 'User Profile.csv')
user_activities = pd.read_csv(path + 'User Activities.csv')

# for new users the articles will be recommended by comparing with old users
# first find the relationship between different users based on euclidean distance

# data processing
# Gender column
user_profiles['Gender_Male'] = np.where(user_profiles['Gender'] == 'Male', 1, 0)
user_profiles.drop(columns = ['Gender'], axis = 1, inplace = True)

# Geography column
user_profiles['Geography_India'] = np.where(user_profiles['Geography'] == 'India', 1, 0)
user_profiles.drop(columns = ['Geography'], axis = 1, inplace = True)

# Profession
user_profiles = pd.get_dummies(user_profiles, columns = ['Profession'], prefix=['Profession'])

# Now we have to find euclidean distance between each user profiles
# Normalize the data frame column 'Age'
scaler = MinMaxScaler(feature_range = (0, 1)) 
user_profiles[['Age']] = scaler.fit_transform(user_profiles[['Age']]) 

# look for profiles whose user activities is present
active_user = user_activities['User Id'].unique()
# find the index of active user
user_idx = np.where(user_profiles['Name'].isin(active_user))[0]
# drop the column 'Name' while finding distance 
data = user_profiles.drop(columns ='Name', axis=1)
dist_mat = squareform(pdist(data, 'euclid'))
# subset the dist_mat by using only user_idx columns
dist_mat_n = dist_mat[:,user_idx]

# Based on distance matrix assign closest member to profiles data frame
user_profiles['Similar User'] = 'X' 
for i in range(len(dist_mat_n)):
    a = np.array(dist_mat_n[i,:])
    minval = np.min(a[np.nonzero(a)])  
    idx = np.where(a == minval)[0][0]
    z = user_profiles.iloc[idx]['Name']
    user_profiles.loc[i, 'Similar User'] = z
    
###### consider a new profile let us say userid = 'C'
U_id = 'F'  
similar_profiles = user_profiles[user_profiles['Name'] == U_id]['Similar User'].values[0]
reco_df = user_activities[user_activities['User Id'] == similar_profiles]
# however, user b might not have liked some articles
# so if a articles time spend is less than 100secs then we will not be suggesting it
reco_df_final = reco_df[reco_df['Time spends (secs)'] > 100]
print('The recommended content id for user', U_id, 'is', reco_df_final['Content Id'].values[0])

##_________________________________________________________________________________________________________
# Now we will recommend content id for Active or Old Users based on content filtering
# We will do content filtering using headlines only
# Note we can also use short description, 'Author and category' and take a weighted average for all the prediction 
# using linear regression on the top of each model.
# However, here we will be taking headline into consideration to keep it simple
# Extract headline and content_id
headlines=[]
content_ids=[]
for data in data_json:
    headline = data['headline']    
    content_id = int(data['content_id'])
    # print(content_id)
    headlines.append(headline)
    content_ids.append(content_id)
    
# natural language processing
lemma = WordNetLemmatizer() # make a object of the Lemmatization class
my_stop = set (stopwords.words('english') + list(punctuation) + ["'" , "'s" , "’", "‘" ])

# tokenize and lemmatize
def tok_lem(message):
    message = message.lower()
    words = word_tokenize(message)
    words_sans_stop = []
    for word in words:
        word = re.sub('[^a-z]+', '', word) # keep only alphabets
        if word in my_stop: # remove the stop words
            continue
        elif len(word) == 0: # remove null words
            continue
        elif word.isdigit(): # remove digits
            continue
        
        words_sans_stop.append(word)
        
    return [lemma.lemmatize(word) for word in words_sans_stop]

words_lists = []

for i in range(len(headlines)):
    words_list = tok_lem(headlines[i])
    words_lists.append(words_list)
    
# let us find the max length
words_len = []
for words in words_lists:
    l = len(words)
    words_len.append(l)
max(words_len) # 12 is the maximum word len

# now we have to decide our length
np.quantile(words_len, 0.95) # which is 12
# so our word length will be 10 here

# build vocabulary
vocab = []
for words in words_lists:
    vocab.extend(words)

# there 811 words
vocab = list(set(vocab)) # 605 unique words

#convert the vocabulary to embeddings
word2vec_model = Word2Vec(words_lists, min_count=1, sg = 1) # create a word to vec model (skip-gram) 
#word_vectors = word2vec_model.wv

def compute_similarity(list_i, list_j):
    sim_score_word_lists = []
    for word_i in list_i:
        sim_score = [word2vec_model.similarity(word_i, word_j) for word_j in list_j]
        #print(sim_score)
        avg_sim_score_word = np.average(sim_score)
        sim_score_word_lists.append(avg_sim_score_word)
        
    avg_sim_score_sent = np.average(sim_score_word_lists)
    return(avg_sim_score_sent)

#compute_similarity(['mass', 'shooting'], ['hugh','grant'])
similarity_matrix = [] # expecting 100*100 matrix
for i,word_list_i in enumerate(words_lists):
    sim_score_sent = []
    for j,word_list_j in enumerate(words_lists):
        if i == j:
            continue
        else:
            sim_score_word_list_i = compute_similarity(word_list_i, word_list_j)
            sim_score_sent.append(sim_score_word_list_i)
    similarity_matrix.append(sim_score_sent)
# so the shape of similarity matrix is 101*100

# now we have suggest a new article to active user
U_Id = 'A'
#find the content_id which user liked
df = user_activities[user_activities['User Id'] == U_Id]
# subset the content which user liked
c_Id = df[df['Time spends (secs)'] > 100]['Content Id'].values[0]

# search for article similar to c_Id
idx = content_ids.index(c_Id)

# retrieve the similarity index
sim = similarity_matrix[idx]

#find the index of highest score
idx2 = np.argmax(sim)

# find the content_id for index
pred_c_Id = content_ids[idx2]
print('The recommended content id for user', U_Id, 'is', pred_c_Id)

# The application should search for whether the user id is for new or old and suggest content accordingly
# I have not built it as a application, it is just and example, hence it is not structured
    
    
    
