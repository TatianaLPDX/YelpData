#!/usr/bin/env python
# coding: utf-8

# # 1. Project Objective 
# Yelp dataset has a great potential for analysis and the pain purpose of this dataset analysis is to - 
# Conduct thorough analysis on 5 different cuisine types of restaurants which are American, Thai, Indian, Japanese and  figure out what makes a good restaurant. Specifically, will mainly analyze customers' reviews and figure out reasons why customers love or dislike the restaurant.

# # 2. Description of DataSet
# The Yelp dataset is downloaded from Kaggle website. This project focuses on two tables which are business table and review table.
# 
# Attributes of business table are as following:
# 
# business_id: ID of the business
# name: name of the business
# neighborhood
# address: address of the business
# city: city of the business
# state: state of the business
# postal_code: postal code of the business
# latitude: latitude of the business
# longitude: longitude of the business
# stars: average rating of the business
# review_count: number of reviews received
# is_open: 1 if the business is open, 0 therwise
# categories: multiple categories of the business
# 
# Attribues of review table are as following:
# 
# review_id: ID of the review
# user_id: ID of the user
# business_id: ID of the business
# stars: ratings of the business
# date: review date
# text: review from the user
# useful: number of users who vote a review as usefull
# funny: number of users who vote a review as funny
# cool: number of users who vote a review as cool

# ### Data preparation

# In[2]:


import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# ### DATA CONVERSION TO CSV, CLEANING AND PLOT FOR YELP BUSINESS

# In[2]:


#convert business json file to pandas df/csv
df = pd.read_json ('business.json',lines=True)


# In[5]:


df.to_csv (r'yelp_business.csv', index = None)


# In[3]:


#read the business details 
business = pd.read_csv('yelp_business.csv')
business.head()


# In[4]:


## remove quotation marks in name and address column
business.name=business.name.str.replace('"','')
business.address=business.address.str.replace('"','')


# In[5]:


## drop all the null values and select only restaurants from Yelp Business Dataset
business = business.dropna()
restaurants = business[business['categories'].str.contains('Restaurants')]


# In[6]:


# Add a category field to restaurants to distinguish the various categories 
## select out 16 cuisine types of restaurants and rename the category
restaurants.is_copy=False
restaurants['category']=pd.Series()
restaurants.loc[restaurants.categories.str.contains('American'),'category'] = 'American'
restaurants.loc[restaurants.categories.str.contains('Mexican'), 'category'] = 'Mexican'
restaurants.loc[restaurants.categories.str.contains('Italian'), 'category'] = 'Italian'
restaurants.loc[restaurants.categories.str.contains('Japanese'), 'category'] = 'Japanese'
restaurants.loc[restaurants.categories.str.contains('Chinese'), 'category'] = 'Chinese'
restaurants.loc[restaurants.categories.str.contains('Thai'), 'category'] = 'Thai'
restaurants.loc[restaurants.categories.str.contains('Mediterranean'), 'category'] = 'Mediterranean'
restaurants.loc[restaurants.categories.str.contains('French'), 'category'] = 'French'
restaurants.loc[restaurants.categories.str.contains('Vietnamese'), 'category'] = 'Vietnamese'
restaurants.loc[restaurants.categories.str.contains('Greek'),'category'] = 'Greek'
restaurants.loc[restaurants.categories.str.contains('Indian'),'category'] = 'Indian'
restaurants.loc[restaurants.categories.str.contains('Korean'),'category'] = 'Korean'
restaurants.loc[restaurants.categories.str.contains('Hawaiian'),'category'] = 'Hawaiian'
restaurants.loc[restaurants.categories.str.contains('African'),'category'] = 'African'
restaurants.loc[restaurants.categories.str.contains('Spanish'),'category'] = 'Spanish'
restaurants.loc[restaurants.categories.str.contains('Middle_eastern'),'category'] = 'Middle_eastern'
restaurants.category[:30]


# In[7]:


## drop null values in category, delete original column categories and reset the index
restaurants=restaurants.dropna(axis=0, subset=['category'])
del restaurants['categories']
restaurants=restaurants.reset_index(drop=True)
restaurants.head(10)


# In[8]:


restaurants.shape


# In[9]:


## check whether has duplicated business id
restaurants.business_id.duplicated().sum()


# In[13]:


plt.style.use('ggplot')


# In[89]:


#Top 10 cities with most restaurants 
grouped = restaurants.city.value_counts()[:10]
plt.figure(figsize=(11,6))
sns.barplot(grouped.index, grouped.values, palette=sns.color_palette("GnBu_r", len(grouped)))
plt.ylabel('Number of restaurants', fontsize=14, labelpad=10)
plt.xlabel('City', fontsize=14, labelpad=10)
plt.title('Count of Restaurants by City (Top 10)', fontsize=15)
plt.tick_params(labelsize=10)
plt.xticks(rotation=30)
for  i, v in enumerate(grouped):
    plt.text(i, v*1, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)


# In[88]:


#Distrubution of restaurant in each state 
plt.figure(figsize=(11,6))
grouped = restaurants.state.value_counts()
sns.barplot(grouped.index, grouped.values,palette=sns.color_palette("GnBu_r", len(grouped)) )
plt.ylabel('Number of restaurants', fontsize=14)
plt.xlabel('State', fontsize=14)
plt.title('Count of Restaurants by State', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=30)
for  i, v in enumerate(grouped):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center', fontweight='bold', fontsize=14)


# DATA CONVERSION TO CSV, CLEANING AND PLOT FOR YELP REVIEWS

# In[ ]:


df_review = pd.read_json ('review.json',lines=True)


# In[ ]:


#convert the reviews to csv 
df_review.to_csv (r'yelp_reviews.csv', index = None)


# In[10]:



#read the review details 
reviews = pd.read_csv('yelp_reviews.csv')
reviews.head()


# Merge two datsets - buisness and reviews 

# In[11]:


## merge business table and review table
restaurants_reviews = pd.merge(restaurants, reviews, on = 'business_id')


# In[87]:


#Top 10 cities with most reviews 
plt.figure(figsize=(11,6))
grouped = restaurants_reviews.groupby('city')['review_count'].sum().sort_values(ascending=False)[:10]
sns.barplot(grouped.index, grouped.values, palette=sns.color_palette("GnBu_r", len(grouped)) )
plt.xlabel('City', labelpad=12, fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Count of Reviews by City (Top 10)', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=15)
for  i, v in enumerate(grouped):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)


# In[22]:


restaurants_reviews.head()


# In[12]:


## update column names
restaurants_reviews.rename(columns={'stars_x':'avg_star','stars_y':'review_star'}, inplace=True)


# In[16]:


restaurants_reviews.head()


# In[13]:


restaurants_reviews['labels'] = ''
restaurants_reviews.loc[restaurants_reviews.review_star >=4, 'labels'] = 'positive'
restaurants_reviews.loc[restaurants_reviews.review_star ==3, 'labels'] = 'neutral'
restaurants_reviews.loc[restaurants_reviews.review_star <3, 'labels'] = 'negative'

# drop neutral reviews for easy analysis
restaurants_reviews.drop(restaurants_reviews[restaurants_reviews['labels'] =='neutral'].index, axis=0, inplace=True)
restaurants_reviews.reset_index(drop=True, inplace=True)

restaurants_reviews.head()


# In[18]:


# distribution of restaurants in each category 
plt.figure(figsize=(11,7))
grouped = restaurants.category.value_counts()
sns.countplot(y='category',data=restaurants, 
              order = grouped.index, palette= sns.color_palette("husl", len(grouped)))
plt.xlabel('Number of restaurants', fontsize=14, labelpad=10)
plt.ylabel('Category', fontsize=14)
plt.title('Count of Restaurants by Category', fontsize=15)
plt.tick_params(labelsize=14)
for  i, v in enumerate(restaurants.category.value_counts()):
    plt.text(v, i+0.15, str(v), fontweight='bold', fontsize=14)


# In[19]:


#Top 10 restaurants with most reviews 
plt.figure(figsize=(11,6))
grouped = restaurants[['name','review_count']].sort_values(by='review_count', ascending=False)[:10]
sns.barplot(x=grouped.review_count, y = grouped.name, palette=sns.color_palette("GnBu_r", len(grouped)), ci=None)
plt.xlabel('Count of Review', labelpad=10, fontsize=14)
plt.ylabel('Restaurants', fontsize=14)
plt.title('TOP 10 Restaurants with Most Reviews', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=15)
for  i, v in enumerate(grouped.review_count):
    plt.text(v, i, str(v), fontweight='bold', fontsize=14)


# ## Review Analysis 

# In[14]:


import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Importing NLTK library for using stop words method
import nltk
from nltk.corpus import stopwords


# In[15]:


## convert text to lower case
restaurants_reviews.text = restaurants_reviews.text.str.lower()

## remove unnecessary punctuation
restaurants_reviews['removed_punct_text']= restaurants_reviews.text.str.replace('\n','').                                           str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','')


# In[16]:


## import positive file which contains common meaningless positive words such as good
file_positive = open('positive.txt')
reader =csv.reader(file_positive)
positive_words = [word[0] for word in reader]


# In[17]:


## import negative file which contains common meaningless positive words such as bad
file_negative = open('negative.txt', encoding = "ISO-8859-1")
reader =csv.reader(file_negative)
negative_words = [word[0] for word in reader]


# In[18]:


## get dataset by category
def get_dataset(category):
    df = restaurants_reviews[['removed_punct_text','labels']][restaurants_reviews.category==category]
    df.reset_index(drop=True, inplace =True)
    df.rename(columns={'removed_punct_text':'text'}, inplace=True)
    return df



## only keep positive and negative words
def filter_words(review):
    words = [word for word in review.split() if word in positive_words + negative_words]
    words = ' '.join(words)
    return words


# In[ ]:


df.to_csv (r'yelp_restaurant_reviews.csv', index = None)


# # Italian Cuisine 

# In[19]:


italian_reviews = get_dataset('Italian')


# In[21]:


italian_train, italian_test = train_test_split(italian_reviews[['text','labels']],test_size=0.5)


# In[22]:


print('Total %d number of reviews' % italian_train.shape[0])


# In[25]:


def split_data(dataset, test_size):
    df_train, df_test = train_test_split(dataset[['text','labels']],test_size=test_size)
    return df_train


# In[26]:


## filter words
italian_train.text = italian_train.text.apply(filter_words)


# In[ ]:


italian_reviews.to_csv(r'yelp_restaurant_reviews_italian.csv', index = None)


# In[27]:



## construct features and labels
terms_train=list(italian_train['text'])
class_train=list(italian_train['labels'])

terms_test=list(italian_test['text'])
class_test=list(italian_test['labels'])


# In[28]:


# get bag of words : the frequencies of various words appeared in each review
vectorizer = CountVectorizer()
feature_train_counts=vectorizer.fit_transform(terms_train)
feature_train_counts.shape


# In[29]:


## run model
svm = LinearSVC()
svm.fit(feature_train_counts, class_train)


# In[ ]:




## create dataframe for score of each word in a review calculated by svm model
coeff = svm.coef_[0]
italian_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})


# In[31]:


## get frequency of each word in all reviews in specific category
italian_reviews = pd.DataFrame(feature_train_counts.toarray(), columns=vectorizer.get_feature_names())
italian_reviews['labels'] = class_train
italian_frequency = italian_reviews[italian_reviews['labels'] =='positive'].sum()[:-1]


# In[32]:


italian_words_score.set_index('word', inplace=True)


# In[34]:


italian_polarity_score = italian_words_score
italian_polarity_score['frequency'] = italian_frequency


# In[35]:


## calculate polarity score 
italian_polarity_score['polarity'] = italian_polarity_score.score * italian_polarity_score.frequency / italian_reviews.shape[0]


# In[36]:


## drop unnecessary words
unuseful_positive_words = italian_polarity_score.loc[['great','amazing','love','best','awesome','excellent','good',
                                                    'favorite','loved','perfect','gem','perfectly','wonderful',
                                                    'happy','enjoyed','nice','well','super','like','better','decent','fine',
                                                    'pretty','enough','excited','impressed','ready','fantastic','glad','right',
                                                    'fabulous']]
unuseful_negative_words =  italian_polarity_score.loc[['bad','disappointed','unfortunately','disappointing','horrible',
                                                     'lacking','terrible','sorry', 'disappoint']]

italian_polarity_score.drop(unuseful_positive_words.index, axis=0, inplace=True)
italian_polarity_score.drop(unuseful_negative_words.index, axis=0, inplace=True)


# In[37]:


italian_polarity_score.polarity = italian_polarity_score.polarity.astype(float)
italian_polarity_score.frequency = italian_polarity_score.frequency.astype(float)


# In[44]:


italian_polarity_score[italian_polarity_score.polarity<0].sort_values('polarity', ascending=True)[:10]


# In[48]:


def plot_top_words(top_words, category):
    plt.figure(figsize=(11,6))
    colors = ['red' if c < 0 else 'blue' for c in top_words.values]
    sns.barplot(y=top_words.index, x=top_words.values, palette=colors)
    plt.xlabel('Polarity Score', labelpad=10, fontsize=14)
    plt.ylabel('Words', fontsize=14)
    plt.title('TOP 10 Positive and Negative Words in %s Restaurants ' % category, fontsize=15)
    plt.tick_params(labelsize=14)
    plt.xticks(rotation=15)


# In[50]:


Italian_top_positive_words = ['delicious','friendly','fresh','recommend','outstanding','incredible','perfection',
                             'attentive','reasonable','phenomenal']
Italian_top_negative_words = ['cold','warm','hard','wrong','bland','slow','wrong','expensive','overpriced','mediocre','poor']
Italian_top_words = italian_polarity_score.loc[Italian_top_positive_words+Italian_top_negative_words,'polarity']
plot_top_words(Italian_top_words,'Italian')


# In[51]:


def get_polarity_score(dataset):
    dataset.text = dataset.text.apply(filter_words)
    
    terms_train=list(dataset['text'])
    class_train=list(dataset['labels'])
    
    ## get bag of words
    vectorizer = CountVectorizer()
    feature_train_counts=vectorizer.fit_transform(terms_train)
    
    ## run model
    svm = LinearSVC()
    svm.fit(feature_train_counts, class_train)
    
    ## create dataframe for score of each word in a review calculated by svm model
    coeff = svm.coef_[0]
    cuisine_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})
    cuisine_reviews = pd.DataFrame(feature_train_counts.toarray(), columns=vectorizer.get_feature_names())
    cuisine_reviews['labels'] = class_train
    cuisine_frequency = cuisine_reviews[cuisine_reviews['labels'] =='positive'].sum()[:-1]
    
    cuisine_words_score.set_index('word', inplace=True)
    cuisine_polarity_score = cuisine_words_score
    cuisine_polarity_score['frequency'] = cuisine_frequency
    
    cuisine_polarity_score.score = cuisine_polarity_score.score.astype(float)
    cuisine_polarity_score.frequency = cuisine_polarity_score.frequency.astype(int)
    
    ## calculate polarity score 
    cuisine_polarity_score['polarity'] = cuisine_polarity_score.score * cuisine_polarity_score.frequency / cuisine_reviews.shape[0]
    
    cuisine_polarity_score.polarity = cuisine_polarity_score.polarity.astype(float)
    ## drop unnecessary words
    unuseful_positive_words = ['great','amazing','love','best','awesome','excellent','good',
                                                   'favorite','loved','perfect','gem','perfectly','wonderful',
                                                    'happy','enjoyed','nice','well','super','like','better','decent','fine',
                                                    'pretty','enough','excited','impressed','ready','fantastic','glad','right',
                                                    'fabulous']
    unuseful_negative_words =  ['bad','disappointed','unfortunately','disappointing','horrible',
                                                    'lacking','terrible','sorry']
    unuseful_words = unuseful_positive_words + unuseful_negative_words
    cuisine_polarity_score.drop(cuisine_polarity_score.loc[unuseful_words].index, axis=0, inplace=True)
    
    return cuisine_polarity_score


# In[55]:


def get_top_words(dataset, label, number=20):
    if label == 'positive':
        df = dataset[dataset.polarity>0].sort_values('polarity',ascending = False)[:number]
    else:
        df = dataset[dataset.polarity<0].sort_values('polarity')[:number]
    return df


# ## MEXICAN CUISINE

# In[52]:


Mexican_reviews = get_dataset('Mexican')
Mexican_train = split_data(Mexican_reviews, 0.7)
print('Total %d number of reviews' % Mexican_train.shape[0])


# In[53]:


Mexican_polarity_score = get_polarity_score(Mexican_train)


# In[59]:


get_top_words(Mexican_polarity_score, 'positive',12)


# In[58]:


get_top_words(Mexican_polarity_score,'negative',10)


# In[60]:


Mexican_top_positive_words = ['delicious','friendly','fresh','recommend','fast','authentic',
                               'incredible','clean','reasonable','fun']
Mexican_top_negative_words = ['bland','cold','slow','wrong','hard','expensive','warm',
                               'greasy','overpriced','mediocre']
Mexican_top_words = Mexican_polarity_score.loc[Mexican_top_positive_words+Mexican_top_negative_words,'polarity']


# In[62]:


plot_top_words(Mexican_top_words,'Mexican')


# In[ ]:





# ## CHINESE CUISINE

# In[63]:


Chinese_reviews = get_dataset('Chinese')
Chinese_train = split_data(Chinese_reviews, 0.7)
print('Total %d number of reviews' % Chinese_train.shape[0])


# In[64]:


Chinese_polarity_score = get_polarity_score(Chinese_train)
get_top_words(Chinese_polarity_score, 'positive',12)


# In[65]:


get_top_words(Chinese_polarity_score,'negative',12)


# In[68]:


Chinese_top_positive_words = ['delicious','fresh','friendly','fast','authentic','hot',
                               'recommend','reasonable','tender','pleasantly']
Chinese_top_negative_words = ['bland','cold','sour','hard','greasy','slow','wrong',
                               'expensive','mediocre','rude']
Chinese_top_words = Chinese_polarity_score.loc[Chinese_top_positive_words+Chinese_top_negative_words,'polarity']


# In[69]:


plot_top_words(Chinese_top_words,'Chinese')


# ## Japanese Cuisine

# In[70]:


Japanese_reviews = get_dataset('Japanese')
Japanese_train = split_data(Japanese_reviews, 0.6)
print('Total %d number of reviews' % Japanese_train.shape[0])


# In[71]:


Japanese_polarity_score = get_polarity_score(Japanese_train)


# In[72]:


get_top_words(Japanese_polarity_score, 'positive',12)


# In[73]:


get_top_words(Japanese_polarity_score,'negative',12)


# In[80]:


Japanese_top_positive_words = ['delicious','fresh','friendly','recommend','reasonable','fast', 'fun',
                               'tender','incredible','variety']
Japanese_top_negative_words = ['slow','cold','bland','wrong','mediocre','expensive','hard'
                               ,'overpriced','poor','warm']
Japanese_top_words = Japanese_polarity_score.loc[Japanese_top_positive_words+Japanese_top_negative_words,'polarity']


# In[81]:


plot_top_words(Japanese_top_words,'Japanese')


# ## Indian Cuisine

# In[74]:


Indian_reviews = get_dataset('Indian')
Indian_train = split_data(Indian_reviews, 0.6)
print('Total %d number of reviews' % Indian_train.shape[0])


# In[75]:


Indian_polarity_score = get_polarity_score(Indian_train)


# In[76]:


get_top_words(Indian_polarity_score, 'positive',12)


# In[77]:


get_top_words(Indian_polarity_score,'negative',12)


# In[84]:


Indian_top_positive_words = ['delicious','friendly','fresh','recommend','attentive','reasonable',
                               'pleasantly','outstanding','authentic','tender']
Indian_top_negative_words = ['cold','bland','sweet','expensive','hard','warm','slow',
                               'problem','greasy','mediocre']
Indian_top_words = Indian_polarity_score.loc[Indian_top_positive_words+Indian_top_negative_words,'polarity']


# In[85]:


plot_top_words(Indian_top_words,'Indian')


# In[ ]:




