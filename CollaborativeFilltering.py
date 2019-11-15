"""
Created on Tue Sep 17 17:01:11 2019

@author: ivona
"""

import pandas as pd
from scipy.spatial.distance import hamming, cosine, correlation
import numpy as np
import sys
import MySQLdb
from collections import Counter
# import mysql.connector
# from mysql.connector import Error
import json

# Current user id
currentUser = int(sys.argv[1])
# number of recommended hashtags
K = int(sys.argv[2])

# Real connection that works on the server
conn = MySQLdb.connect(host='localhost', user='root', passwd='123123', db='hashtags')

# Connection for testing
# conn = mysql.connector.connect(host='localhost',
#                                         database='hashtags',
#                                         user='root',
#                                         password='123123')

# Define liked_hashtags Dataframe
cur = conn.cursor()
cur.execute("SHOW columns FROM liked_hashtags")
columns_liked = [column[0] for column in cur.fetchall()]
cur.execute("SELECT * FROM liked_hashtags")
record_liked_hashtags = cur.fetchall()
liked_hashtags = pd.DataFrame(record_liked_hashtags, columns = columns_liked)
liked_hashtags = liked_hashtags[['id', 'name', 'avg_likes', 'main_id']]

# Define main_hashtags DataFrame
cur.execute("SHOW columns FROM main_hashtags")
columns_main = [column[0] for column in cur.fetchall()]
cur.execute("SELECT * FROM main_hashtags")
record_main_hashtags = cur.fetchall()
main_hashtags = pd.DataFrame(record_main_hashtags, columns = columns_main)
main_hashtags = main_hashtags[['id', 'name', 'total_counter_search', 'related_id']]
main_hashtags.set_index('id', inplace=True)

# Define related_hashtags DataFrame
cur.execute("SHOW columns FROM related_hashtags")
columns_related = [column[0] for column in cur.fetchall()]
cur.execute("SELECT * FROM related_hashtags")
record_related_hashtags = cur.fetchall()
related_hashtags = pd.DataFrame(record_related_hashtags, columns = columns_related)
related_hashtags = related_hashtags[['id', 'name']]
related_hashtags.set_index('id', inplace=True)

# Define user_favorite_hashtags DataFrame
cur.execute("SHOW columns FROM user_favorite_hashtags")
columns_user_fave = [column[0] for column in cur.fetchall()]
cur.execute("SELECT * FROM user_favorite_hashtags")
record_user_favorite_hashtags = cur.fetchall()
user_favorite_hashtags = pd.DataFrame(record_user_favorite_hashtags, columns = columns_user_fave)
user_favorite_hashtags = user_favorite_hashtags[["id", "user_id", "category", "hashtags"]]

# Define user_main_hashtags DataFrame
cur.execute("SHOW columns FROM user_main_hashtags")
columns_user_main = [column[0] for column in cur.fetchall()]
cur.execute("SELECT * FROM user_main_hashtags")
record_user_main_hashtags = cur.fetchall()
user_main_hashtags = pd.DataFrame(record_user_main_hashtags, columns = columns_user_main)
user_main_hashtags = user_main_hashtags[["id", "user_id", "main_id", "counter_search"]]

conn.close()

usersPerHashtag = user_main_hashtags.main_id.value_counts()

hashtagsPerUser = user_main_hashtags.user_id.value_counts()

user_main_hashtags = user_main_hashtags[user_main_hashtags["main_id"].isin(usersPerHashtag[usersPerHashtag>0].index)]
user_main_hashtags = user_main_hashtags[user_main_hashtags["user_id"].isin(hashtagsPerUser[hashtagsPerUser>0].index)]
user_main_hashtags["counter_search"] = pd.to_numeric(user_main_hashtags["counter_search"])

# Creating a pivot table to see which user has searched which hashtag
userHashtagsMatrix = pd.pivot_table(user_main_hashtags, values = 'counter_search', index = ['user_id'], columns = ['main_id'])
userHashtagsMatrix = userHashtagsMatrix.fillna(0)



# User Liked Hashtags
userLikesHashtags = []

for index, row in user_favorite_hashtags.iterrows():
    hashtagsArray = str(row["hashtags"]).replace("#", "").replace(" ", "").split(",")
    for hashtag in hashtagsArray:
        userLikesHashtags.append([row["user_id"], hashtag, 10])
    
userLikedHashtagsDf = pd.DataFrame(userLikesHashtags, columns=['user_id', 'hashtag', 'rating'])

userLikedHashtagsMatrix = pd.pivot_table(userLikedHashtagsDf, values = 'rating', index = ['user_id'], columns = ['hashtag'])    

# Get the hashtag and the related hashtags by id
def getHashtag(id):
    hashtag = main_hashtags.at[id, "name"]
    relatedHashtagId = main_hashtags.at[id, "related_id"]
    relatedHashtags = related_hashtags.at[relatedHashtagId, "name"]
    return hashtag

# Get User Favorite Hashtags (he has searched the most)
def faveHashtags(user, N):
    userHashtagSearches = user_main_hashtags[user_main_hashtags["user_id"] == user]
    sortedSearches = pd.DataFrame.sort_values(userHashtagSearches, ['counter_search'], ascending=[0])[:N]
    sortedSearches['hashtag'] = sortedSearches['main_id'].apply(getHashtag)
    
    allHashtags = []
    userHashtagLiked = user_favorite_hashtags[user_favorite_hashtags["user_id"] == user]
    for hashtag in userHashtagLiked:
        hashtags = userHashtagLiked["hashtags"]
        allHashtags.extend(hashtags.values[0].split(","))
    uniqueHashtag = Counter(allHashtags).keys()
    numberOfAppearences = Counter(allHashtags).values()  
    uniqueHashtagsCount = np.array([list(uniqueHashtag), list(numberOfAppearences)]).T
    uniqueHashtagsCountTable = pd.DataFrame(uniqueHashtagsCount, columns = ["Hashtag", "Count"])
    uniqueHashtagsCountTable.sort_values("Count", ascending=False)
    
    return sortedSearches, uniqueHashtagsCountTable["Hashtag"]

def distance(user1, user2, userMatrix, distanceFunction = "hamming"):
    try:
        user1Ratings = userMatrix.transpose()[user1]
        user2Ratings = userMatrix.transpose()[user2]
        if distanceFunction == "correlation":
            distance = correlation(user1Ratings,user2Ratings)
        elif distanceFunction == "hamming":
            distance = hamming(user1Ratings,user2Ratings)
        else:
            distance = np.NaN
    except: 
        distance = np.NaN
    return distance

def nearestNeighborsMostSearched(user, K=10):
    allUsers = pd.DataFrame(userHashtagsMatrix.index)
    allUsers = allUsers[allUsers.user_id!=user]
    allUsers["distance"] = allUsers["user_id"].apply(lambda x: distance(user,x, userHashtagsMatrix, "correlation"))
    KnearestUsers = allUsers.sort_values(["distance"],ascending=True)["user_id"][:K]
    return KnearestUsers

def nearestNeighborsLiked(user, K=10):
    allUsers = pd.DataFrame(userLikedHashtagsMatrix.index)
    allUsers = allUsers[allUsers.user_id!=user]
    allUsers["distance"] = allUsers["user_id"].apply(lambda x: distance(user,x, userLikedHashtagsMatrix, "hamming"))
    KnearestUsers = allUsers.sort_values(["distance"],ascending=True)["user_id"][:K]
    return KnearestUsers

def topNSearched(user, N=3):
    KnearestUsers = nearestNeighborsMostSearched(user)
    NNRatings = userHashtagsMatrix[userHashtagsMatrix.index.isin(KnearestUsers)]
    avgSearch = NNRatings.apply(np.nanmean).dropna()
    userHashtagsMatrix.replace(0, np.nan, inplace=True)
    hashtagsAlreadySearched = userHashtagsMatrix.transpose()[user].dropna().index
    avgSearch = avgSearch[~avgSearch.index.isin(hashtagsAlreadySearched)]
    topHashtags = avgSearch.sort_values(ascending=False).index[:N]
    return pd.Series(topHashtags).apply(getHashtag)
    
def topNLiked(user,N=3):
    KnearestUsers = nearestNeighborsLiked(user)
    NNRatings = userLikedHashtagsMatrix[userLikedHashtagsMatrix.index.isin(KnearestUsers)]
    avgSearch = NNRatings.apply(np.nanmean).dropna()
    hashtagsAlreadySearched = userLikedHashtagsMatrix.transpose()[user].dropna().index
    avgSearch = avgSearch[~avgSearch.index.isin(hashtagsAlreadySearched)]
    topHashtags = avgSearch.sort_values(ascending=False).index[:N]
    return pd.Series(topHashtags)


fave, liked = faveHashtags(currentUser, K)
recommendedHashtagsBySearch = topNSearched(currentUser, K)
recommendedHashtagsByLikes = topNLiked(currentUser, K)
res = {"by_search" : recommendedHashtagsBySearch.values.tolist(), "by_likes" : recommendedHashtagsByLikes.values.tolist()}
print(json.dumps(res))