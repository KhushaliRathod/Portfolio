#!/usr/bin/env python
# coding: utf-8

# ## Book Recommendation System

# ### Used clustering

# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


# In[8]:


books = pd.read_csv('Book Dataset/books.csv' , sep = ';' , on_bad_lines='skip' ,encoding='latin-1')


# In[9]:


books.head()


# In[11]:


books.isna().sum()


# In[13]:


books.shape


# In[14]:


books.columns


# In[15]:


books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]


# In[16]:


books.head()


# In[17]:


books.rename (columns = {
    'Book-Title' : 'Title' ,
    'Book-Author' : 'Author' ,
    'Year-Of-Publication' : 'Year',
    'Publisher' : 'Publisher',
    'Image-URL-L' : 'IMG_URL'},inplace = True )


# In[18]:


books.head(2)


# In[21]:


users = pd.read_csv('Book Dataset/users.csv' , sep = ';' , on_bad_lines='skip' ,encoding='latin-1')


# In[22]:


users.head(2)


# In[23]:


users.shape


# In[26]:


users.isna().sum()


# In[27]:


ratings= pd.read_csv('Book Dataset/ratings.csv' , sep = ';' , on_bad_lines='skip' ,encoding='latin-1')


# In[28]:


ratings.head(2)


# In[29]:


ratings.shape


# In[30]:


ratings.isna().sum()


# In[31]:


# To find how many user has provided rating.
ratings['User-ID'].value_counts()


# In[35]:


# To find unique user
ratings['User-ID'].unique().shape


# In[37]:


x = ratings['User-ID'].value_counts() > 200


# In[39]:


x[x].shape


# In[40]:


y = x[x].index


# In[43]:


y


# In[45]:


# User who has rating more than 200 books
ratings = ratings[ratings['User-ID'].isin(y)]


# In[46]:


ratings.head()


# In[47]:


ratings.shape


# In[48]:


# Merging of books with ratings
rwb=ratings.merge(books , on = 'ISBN')


# In[49]:


rwb.head()


# In[50]:


rwb.shape


# In[51]:


#Which books get what rating

num_rating = rwb.groupby('Title') ['Book-Rating'].count().reset_index()


# In[52]:


num_rating.head()


# In[55]:


num_rating[num_rating['Book-Rating'] > 5]


# In[57]:


num_rating['Book-Rating'].nlargest()


# In[59]:


final_df = rwb.merge(num_rating , on ='Title')


# In[60]:


final_df.head()


# In[62]:


final_df.rename(columns={
    'Book-Rating_x' : 'Rating' ,
    'Book-Rating_y' : 'Num_Of_Voter'
},inplace = True)


# In[63]:


final_df.head(2)


# In[99]:


final_50=final_df[final_df['Num_Of_Voter'] >=50]


# In[100]:


final_50.head(2)


# In[101]:


a=final_50[final_50.duplicated(subset='Title')]


# In[102]:


a.head(4)


# In[103]:


a.shape


# In[104]:


final_50.drop_duplicates(['User-ID','Title'] , inplace = True)


# In[105]:


final_50.shape


# In[106]:


#Pivot table to build matrix
book_pivot = final_50.pivot_table(columns = 'User-ID' , index = 'Title' , values = 'Rating' )


# In[107]:


book_pivot


# In[108]:


book_pivot.fillna(0 , inplace = True)


# In[109]:


book_pivot


# ### Sparse Matrix
# 
# If most of the elements of the matrix have 0 value, then it is called a sparse matrix. The two major benefits of using sparse matrix instead of a simple matrix are:
# Storage: There are lesser non-zero elements than zeros and thus lesser memory can be used to store only those elements.
# Computing time: Computing time can be saved by logically designing a data structure traversing only non-zero elements.
# 

# In[111]:


#Applying clustering of similar rating in one cluster. 
from scipy.sparse import csr_matrix


# In[112]:


book_sparse = csr_matrix(book_pivot)


# In[113]:


book_sparse


# Brute force
# This algorithm calculates the distance between a query point and every data point in a dataset. It's a general problem-solving technique that's computationally expensive, especially for large datasets. The time complexity of brute force algorithms is generally high, typically O(n!) or O(2^n). 
#  
# KD tree
# This data structure organizes data points in a hierarchical structure to make nearest neighbor searches more efficient. KD trees are well-suited for data that's relatively static or changes infrequently, and are useful for image processing, computer graphics, and geographic information systems (GIS). 
#  
# Ball tree
# This algorithm is similar to KD trees, but uses hyper-spheres (balls) instead of boxes. Ball trees are effective for scenarios where non-Euclidean distances are important, and the dataset has varying spatial densities. They're useful for machine learning with non-Euclidean metrics, document clustering, and molecular biology. 

# In[115]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm = 'brute')


# In[116]:


model.fit(book_sparse)


# ### Meaning of folllowing line
# 
# book_pivot.iloc[230, :]: This part selects the 230th row from the book_pivot DataFrame.
# .values: Converts the selected row into a NumPy array.
# .reshape(1, -1): Reshapes the array into a 2D array with 1 row and an unknown number of columns (determined automatically). This is necessary for the kneighbors method.
# n_neighbors=6: This argument specifies that you want to find the 6 nearest neighbors to the given data point.
# 
# ### Purpose:
# 
# The code is essentially finding the 6 books that are most similar to the book at index 230 in the book_pivot DataFrame. The model object, presumably a K-Nearest Neighbors model, is trained on a dataset of books and their features. By finding the nearest neighbors, you can recommend books that are similar in terms of those features.
# 
# ### Output:
# 
# distance: A NumPy array containing the distances between the query point (book at index 230) and the 6 nearest neighbors.
# suggestion: A NumPy array containing the indices of the 6 nearest neighbors in the original dataset.

# In[123]:


distance , suggestion  = model.kneighbors(book_pivot.iloc[232 , :].values.reshape(1,-1) , n_neighbors = 6)


# In[124]:


suggestion


# In[125]:


for i in range(len(suggestion)):
    print(book_pivot.index[suggestion[i]])


# In[126]:


books_name = book_pivot.index


# In[127]:


import pickle
pickle.dump(model, open('artifacts/model.pkl' , 'wb'))
pickle.dump(books_name, open('artifacts/books_name.pkl' , 'wb'))
pickle.dump(final_df, open('artifacts/final_df.pkl' , 'wb'))
pickle.dump(book_pivot, open('artifacts/book_pivot.pkl', 'wb'))


# In[136]:


def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance , suggestion  = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1,-1) , n_neighbors = 6)
    
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            print(j)


# In[139]:


book_name = 'A Bend in the Road'
recommend_book(book_name)


# In[ ]:




