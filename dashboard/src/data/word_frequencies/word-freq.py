from os import walk
import sys
import pandas as pd
from collections import Counter

CLEAR_PATH = r'../clear/'
FREQ_PATH = r'../word_frequencies/'



def get_corpus_df(df:pd.DataFrame) -> list:
    """
    Reciwes df and returns a list with words that are in the tweets.
    """
    words = []
    
    for row in df:
        for word in eval(row):
            words.append(word)

    return words


def get_corpus_str(text: str) -> list:
    """
    Reciwes str and returns a list with words that are in the tweets.
    """
    words = []
    for i in text:
        i = str(i)
        for j in i.split():
            words.append(j.strip())
    return words


# Get names of files in CLEAR_PATH.
for (dirpath, dirnames, filenames) in walk(CLEAR_PATH):

    # outputs all possible filenames from CLEAR_PATH folder
    if(len(filenames) != 0):
        print('The folowing files were found:')
        print(filenames)
    
    # if there are no files in the CLEAR_PATH folder, scrypt terminates
    else:
        print(f"There are no files in the {CLEAR_PATH} path. ;(")
        print("Exit...\n")
        sys.exit(0)

# Try to read file. 
while True:
    CLEAR_NAME = input("Plese copy and enter the name of the file(enter *exit* if you want to close scrypt): ")

    # You can close script if enter *exit* in console.
    if(CLEAR_NAME == 'exit'):
        print("Exit...\n")
        sys.exit(0)
    
    # If you write a valid name of file, data will write to df variable.
    try:
        df = pd.read_csv(CLEAR_PATH+CLEAR_NAME, sep='|')
        break

    # Otherwise, the program gives you a second chance.
    except FileNotFoundError:
        continue


# Try to read column.
while True:
    list_columns = list(df.columns[3:])

    print(list_columns)

    column = input('Plese copy and enter the name of the column(enter *exit* if you want to close scrypt): ')

    # You can close script if enter *exit* in console.
    if(column == 'exit'):
        print("Exit...\n")
        sys.exit(0)

    # if name wrong program gives you a second chance.
    if (column not in list_columns):
        print("Wrong name of column, try again.")
        continue

    # Read as df if ok.
    try:
        eval(df[column][0])
        corpus = get_corpus_df(df[column])
    
    # read as str if not ok.
    except SyntaxError:
        corpus = get_corpus_str(df[column])

    break


# Word frequency counter in tweets.
counter = Counter(corpus)

# Convert to DF and add name to columns.
freq_df =  pd.DataFrame.from_dict(counter, orient='index')
freq_df['word'] = freq_df.index
freq_df['freq'] = freq_df[0]
freq_df.drop(0, axis=1, inplace=True)
freq_df.reset_index(drop=True, inplace=True)

# Sort by column *freq* in descending order(From larger to smaller).
freq_df.sort_values('freq', inplace=True, ascending=[False])

# Save 2 columns(word, freq) df with *,* separator. Save to FREQ_PATH folder.
FREQ_NAME = f'word-freq_{column}.csv'
freq_df.to_csv(FREQ_PATH+FREQ_NAME, index=False)


