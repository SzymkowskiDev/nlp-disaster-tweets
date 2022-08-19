import re
from itertools import groupby
import emoji
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn

def preprocess_data(frame: pd.DataFrame, options: list):
    """_summary_

    Args:
        frame (pd.DataFrame): _description_
        options (list): _description_

    Returns:
        _type_: _description_
    """
    #sanity check
    if options is None:
        return frame
    
    # Remove blank rows if any.
    frame['text'].dropna(inplace=True)

    # Change all the text to lower case.
    frame['text'] = [entry.lower() for entry in frame['text']]

    frame['text'] = frame['text'].apply(preprocess_text, options=options)
    
    if 11 in options:
        #Tokenize & Lemmatize
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')
        try:
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')

        
        frame['text']= [word_tokenize(entry) for entry in frame['text']]
    
        # Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        tag_map = defaultdict(lambda : wn.NOUN)

        if 12 not in options:
            # 12 option == Leave only nouns
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV

        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()

        for index,entry in enumerate(frame['text']):

            # Declaring Empty List to store the words that follow the rules for this step
            Final_words = []
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
            frame.loc[index,'text'] = str(Final_words)
    
    if 12 in options:
        # TODO: complete this "Leave only nouns" 
        pass

    if 13 in options:
        # TODO: complete this "Spell check"
        pass
    

    return frame

def preprocess_text(text: str, options: list[int]):
    if 1 in options:
        # Remove hashtag while keeping hashtag text
        text = re.sub(r'#','', text)
    
    if 2 in options:
        # Remove HTML special entities (e.g. &amp;)
        text = re.sub(r'\&\w*;', '', text)
    
    if 3 in options:
        # Remove tickers
        text = re.sub(r'\$\w*', '', text)

    if 4 in options:
        # Remove hyperlinks
        text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    
    if 5 in options:
        # Remove whitespace (including new line characters)
        text = re.sub(r'\s\s+','', text)
        text = re.sub(r'[ ]{2, }',' ',text)
    
    if 6 in options:
        # Remove URL, RT, mention(@)
        text=  re.sub(r'http(\S)+', '',text)
        text=  re.sub(r'http ...', '',text)
        text=  re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+','',text)
        text=  re.sub(r'RT[ ]?@','',text)
        text = re.sub(r'@[\S]+','',text)
        #&, < and >
        text = re.sub(r'&amp;?', 'and',text)
        text = re.sub(r'&lt;','<',text)
        text = re.sub(r'&gt;','>',text)
    
    if 7 in options:
        # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
        text= ''.join(c for c in text if c <= '\uFFFF') 
        text = text.strip()
    
    if 8 in options:
        # Remove misspelling words
        text = ''.join(''.join(s)[:2] for _, s in groupby(text))
    
    if 9 in options:
        # Remove emoji
        text = emoji.demojize(text)
        text = text.replace(":"," ")
        text = ' '.join(text.split()) 
        text = re.sub("([^\x00-\x7F])+"," ",text)

    if 10 in options:
        # Remove Mojibake (also extra spaces)
        text = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    
    return text