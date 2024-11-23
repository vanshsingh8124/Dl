#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np


# In[50]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[51]:


movies.head(1)


# In[52]:


movies = movies.merge(credits,on='title')


# In[53]:


movies.head(1)


# In[54]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[55]:


movies.head()


# In[56]:


movies.isnull().sum()


# In[57]:


movies.dropna(inplace=True)


# In[58]:


movies.isnull().sum()


# In[59]:


movies.duplicated().sum()


# In[60]:


movies.iloc[0].genres


# In[61]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[62]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[63]:


movies['genres'] = movies['genres'].apply(convert)


# In[64]:


movies.head()


# In[65]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[66]:


movies.head()


# In[67]:


def convert4(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[68]:


movies['cast'].apply(convert4)


# In[69]:


movies['cast'] = movies['cast'].apply(convert4)


# In[70]:


movies.head()


# In[71]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[72]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[73]:


movies.head()


# In[74]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[75]:


movies.head()


# In[76]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[77]:


movies.head()


# In[78]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[79]:


movies.head()


# In[80]:


new_df = movies[['movie_id','title','tags']]


# In[81]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[82]:


new_df.head()


# In[83]:


new_df['tags'][0]


# In[84]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[85]:


new_df.head()


# In[86]:


pip install -U scikit-learn


# In[87]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[88]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[89]:


vectors[0]


# In[90]:


cv.get_feature_names_out()


# In[91]:


import nltk


# In[92]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer


# In[93]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)
    
        
    


# In[94]:


new_df['tags'][0]


# In[95]:


ps.stem('dispatched')


# In[96]:


from nltk.stem import PorterStemmer

ps = PorterStemmer()
word = 'danced'
stemmed_word = ps.stem(word)
print(stemmed_word)


# In[97]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[98]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[99]:


from sklearn.metrics.pairwise import cosine_similarity


# In[100]:


similarity = cosine_similarity(vectors)


# In[101]:


sorted(list(enumerate(similarity[0])),reverse = True,key=lambda x:x[1])[1:6]


# In[102]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    
    


# In[103]:


recommend('Avatar')


# In[104]:


import pickle


# In[105]:


new_df.to_dict()


# In[106]:


new_df


# In[107]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[108]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




