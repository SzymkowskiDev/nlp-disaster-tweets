
import pandas as pd
import json
import re, itertools, emoji
from wordcloud import STOPWORDS as stop
import string
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
from nltk.stem import WordNetLemmatizer
from datetime import datetime


################################Auxiliary functions###################################

def remove_contractions(text: str) -> str:
    """
    Function to replace contractions with their longer forms

    Args:
    string text: text to replace contractions

    Returns:
    string: replaced text
    """
    new_test = []
    for t in text.split():
        if t.lower() in contractions.keys():
            new_test.append(contractions[t.lower()])
        else:
            new_test.append(t)

    ## TODO: ????
    # assert 'contractions' in globals(), "Json file with contractions not loaded"
    # return contractions[text.lower()] if text.lower() in contractions.keys() else text

    return ' '.join(new_test)


def clean_dataset(text: str) -> str:
    """
    Function to get rif off unwanted patterns
    Args:
    string text: text to clean

    Returns:
    string: replaced text
    """
    # Remove hashtag while keeping hashtag text
    text = re.sub(r'#','', text)
    # Remove HTML special entities (e.g. &amp;)
    text = re.sub(r'\&\w*;', '', text)
    # Remove tickers
    text = re.sub(r'\$\w*', '', text)
    # Remove hyperlinks
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    # Remove whitespace (including new line characters)
    text = re.sub(r'\s\s+','', text)
    text = re.sub(r'[ ]{2, }',' ',text)
    # Remove URL, RT, mention(@)
    text=  re.sub(r'http(\S)+', '',text)
    text=  re.sub(r'http ...', '',text)
    text=  re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+','',text)
    text=  re.sub(r'RT[ ]?@','',text)
    text = re.sub(r'@[\S]+','',text)

    # TODO: why?? example id:13
    #Remove words with 4 or fewer letters
    #text = re.sub(r'\b\w{1,4}\b', '', text)


    #&, < and >
    text = re.sub(r'&amp;?', 'and',text)
    text = re.sub(r'&lt;','<',text)
    text = re.sub(r'&gt;','>',text)
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    text= ''.join(c for c in text if c <= '\uFFFF')
    text = text.strip()
    # Remove misspelling words
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))

    # TODO:
    # Remove emoji
    text = emoji.demojize(text)
    text = text.replace(":"," ")
    text = ' '.join(text.split())
    text = re.sub("([^\x00-\x7F])+"," ",text)
    # Remove Mojibake (also extra spaces)
    text = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    return text


def remove_stopwords(text: str) -> str:
    """

    """
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip().replace('.',' ').replace(',',' '))
    return " ".join(final_text)


def lemmatization(words: list) -> list:
    new_list = []
    l = WordNetLemmatizer()

    for w in words:
        new_list.append(l.lemmatize(w))

    return new_list


######################################PATHS###########################################

ORIGINAL_PATH = r"data/original/"
CLEAR_PATH = r"data/clear/"

##################################Auxiliary files#####################################

# read dictionary with contractions
with open('notebooks\en_contractions.json') as file:
    contractions = json.load(file)


# A collection of words and punctuation marks to remove from tweets
punctuation = list(string.punctuation)
stop.update(punctuation)

nltk.download('averaged_perceptron_tagger')

# add ectra stop words
extra_stopwords = ['s', 'u', 'new', 'will', 'one','2']
stop.update(extra_stopwords)


####################################CLEANING##########################################


# read
df = pd.read_csv(ORIGINAL_PATH + r"train.csv")

# drop na
df['text'].dropna(inplace=True)

# to lower case 
df['lower_text'] = [entry.lower() for entry in df['text']]

# remove cotractions
df['without_contractions']=df['lower_text'].apply(remove_contractions)

# clean from noise
df['without_noise'] = df['without_contractions'].apply(clean_dataset)

# del stopwords
df['without_stopwords'] = df['without_noise'].apply(remove_stopwords)

# tokenization
df['tokenized'] = df['without_stopwords'].apply(word_tokenize)

# adds part of speech
df['pos_tags'] = df['tokenized'].apply(pos_tag)

# lemmantization
df['lemmatized'] = df['tokenized'].apply(lemmatization)

print(df.head(15))

####################################SAVE################################################

NAME = f"clear_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.csv"
df.to_csv(CLEAR_PATH+NAME, encoding='utf-8', index=False, sep='|')