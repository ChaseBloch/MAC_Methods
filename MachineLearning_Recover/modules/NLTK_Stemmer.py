# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:37:34 2022

@author: chase
"""
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords 

#Create TF-IDF stemmer.
stemmer = nltk.stem.PorterStemmer()
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
#Create count stemmer.  
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    

def ProperNounExtractor(text):
    output = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words=[word for word in words if word.isalpha() if word not in set(stopwords.words('english'))]
        tagged = nltk.pos_tag(words)
        for i in range(len(tagged)):
            if tagged[i][1] == 'NNP': # If the word is a proper noun
                output.append(tagged[i][0])
        for i in range(len(tagged)-1):
            if (tagged[i][1] == 'NNP') & (tagged[i+1][1] == 'NNP'):
                output.append(tagged[i][0] + ' ' + tagged[i+1][0])
    return(output)
