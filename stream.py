import streamlit as st
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
import unicodedata
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import unicodedata
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import os
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag,pos_tag_sents
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD as svd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin 
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(f'<h4 style="color:#FFF;font-size:36px;">{"Uber Analytics"}</h4>', unsafe_allow_html=True)

def main():
  # st.title('Explore a dataset')
  st.write('A general purpose data exploration app')
  file = st.sidebar.file_uploader("Upload file", type=['csv', 
                                               'xlsx', 
                                               'pickle'])
  if not file:
    st.sidebar.write("Upload a .csv or .xlsx file to get started")
    return
  df = get_df(file)
  task = st.sidebar.radio('Task', ['Explore', 'Transform'], 0)
  if task == 'Explore':
    explore(df)
  else:
    dd=transform(df)
    # st.write(dd)


data = st.sidebar.file_uploader("upload file here", type = ['csv'])
if data is not None:
    st.markdown(f'<h3 style="color:#008B8B;font-size:34px;">{"Dataset"}</h3>', unsafe_allow_html=True)
    reviews = pd.read_csv(data, encoding='cp1252',)
    st.dataframe(reviews)
    reviews['Date'] = pd.to_datetime(reviews['Date'])
    reviews.drop_duplicates(subset=None,keep='first',inplace=True)
    
    

# Combining title and reviews columns
    reviews['full_review'] = reviews['Title'] + " " + reviews['Review']
# reviews["full_review"] =  reviews.full_review.str.replace('[^\x00-\x7F]','')

    st.write(reviews.full_review)

    def preprocessing(text):
        stop = nltk.corpus.stopwords.words('english')
        lem = WordNetLemmatizer()
        text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
        .decode('utf-8', 'ignore')
        .lower())
        words = re.sub(r'[^\w\s]', '', text).split()
        return [lem.lemmatize(w) for w in words if w not in stop]

    reviews['words']=reviews.apply(lambda x: preprocessing(x['full_review']), axis=1)
    def final(lem_col):
        return (" ".join(lem_col))

    reviews['words'] = reviews.apply(lambda x: final(x['words']),axis=1)
    #st.markdown(f'<h1 style="color:#D2691E;font-size:34px;">{"Review Words are as below "}</h1>', unsafe_allow_html=True)
    #reviews['words']

    words = preprocessing(''.join(str(reviews['full_review'].tolist())))
    

# Bigrams

    bigram = (pd.Series(nltk.ngrams(words,2)).value_counts())[:13]
    st.markdown(f'<h2 style="color:#FFF;font-size:34px;">{"Bigrams"}</h2>', unsafe_allow_html=True)
    st.write(bigram)
    st.markdown(f'<h2 style="color:#FFF;font-size:34px;">{"Barplot of Bigrams"}</h2>', unsafe_allow_html=True)
    plt.figure(figsize=(10,6)) 
    sns.barplot(bigram.values,bigram.index,orient="h",color="#1E90FF")
    plt.show()
    st.pyplot()
    
    
# Trigrams

    trigram = (pd.Series(nltk.ngrams(words,3)).value_counts())[:10]
    st.markdown(f'<h2 style="color:#FFF;font-size:34px;">{"Trigrams"}</h2>', unsafe_allow_html=True)
    st.write(trigram)

#trigram.plot.barh()
    st.markdown(f'<h2 style="color:#FFF;font-size:34px;">{"Barplot of Trigrams"}</h2>', unsafe_allow_html=True)
    sns.barplot(trigram.values,trigram.index,orient="h",color="#FF4500")
    plt.show()
    st.pyplot()


# Wordcloud for Rating greater than or equal to 4

    high_rating = reviews[reviews['Rating'] >= 4]
    stop_words = ["Uber", "app",'driver'] + list(STOPWORDS)

    text = high_rating.full_review.tolist() 
    text = ' '.join(text)
    st.markdown(f'<h2 style="color:#FFF;font-size:34px;">{"Word Cloud of High Rating >= 4 "}</h2>', unsafe_allow_html=True)
    wordcloud = WordCloud(stopwords = stop_words,background_color='white',width= 3000, height = 2000, collocations=True).generate(text)
    plt.figure(figsize=(10,6)) 
    plt.imshow(wordcloud, interpolation='bilInear')
    plt.axis('off')
    plt.show()
    st.pyplot()

# Wordcloud for Rating less than or equal to 3

    low_rating = reviews[reviews['Rating'] <= 3]
    stop_words = ["Uber", "app",'driver'] + list(STOPWORDS)

    text = low_rating.full_review.tolist() 
    text = ' '.join(text)
    st.markdown(f'<h2 style="color:#FFF;font-size:34px;">{"Word Cloud of low Rating <= 3 "}</h2>', unsafe_allow_html=True)
    wordcloud = WordCloud(stopwords = stop_words,background_color='white',width= 3000, height = 2000, collocations=True).generate(text)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilInear')
    plt.axis('off')
    plt.show()
    st.pyplot()

# Supervised learning phase

    
    reviews['binary'] = np.where(reviews['Rating'] >= 3, 1, 0)
    

    count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,1))
    vectorizer = count_vectorizer.fit_transform(reviews.words)

    tfidf_transformer = TfidfTransformer()
    tf_transformer = TfidfTransformer(use_idf=False).fit_transform(vectorizer)

    y = reviews['binary']
    X_train,X_test, y_train, y_test = train_test_split(tf_transformer,y,test_size=0.25,random_state=52)

    mlp_classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

    # Fit the classifier to the training data
    mlp_classifier.fit(X_train,y_train)

    # Create the predicted tags: pred
    pred = mlp_classifier.predict(X_test)
    

    from sklearn import metrics

# Calculate the confusion matrix: cm

    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements
    cm = metrics.confusion_matrix(y_test,pred,labels=[0,1])
    a= accuracy(cm)
    # st.markdown(f'<h2 style="color:#FFF;font-size:25px;">{"Accuracy of Classifier model is as below "}</h2>', unsafe_allow_html=True)
    # st.write(a)

# Logistic regression

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred2 = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test,pred2,labels=[0,1])
    b=accuracy(cm)
    # st.markdown(f'<h1 style="color:#556B2F;font-size:25px;">{"Accuracy of Logistic Regression Model is as below "}</h1>', unsafe_allow_html=True)
    # st.write(b)



# creating array variable of all the words

    feature_names = np.array(count_vectorizer.get_feature_names())

#creating array of all the regression coefficients per word

    coef_index = model.coef_[0]

#creating df with both arrays in it

    df = pd.DataFrame({'Word':feature_names, 'Coef': coef_index})

#sorting by coefficient

    st.markdown(f'<h2 style="color:#FFF;font-size:30px;">{"Logistic Regression: Positive Words "}</h2>', unsafe_allow_html=True)
    st.write(df.sort_values('Coef',ascending= False)[:10])
    st.write(" ")
    st.markdown(f'<h2 style="color:#FFF;font-size:30px;">{"Logistic Regression: Negative Words"}</h2>', unsafe_allow_html=True)
    st.write(df.sort_values('Coef',ascending= True)[:10])




