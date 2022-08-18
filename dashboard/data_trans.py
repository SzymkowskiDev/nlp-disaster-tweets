# DATA TRANSFORMATIONS TO FEED MAP

import pandas as pd

data = pd.read_csv("data/nloc_train.csv")


# PART 1: Getting disaster totals

# print(data.count())
# # keyword       7552
# # location      5080
# # country       3481 (So, 3481 out of 5080)
# # text          7613
# # target        7613

# # Subset only target = 1 (Only disasters)
# data = data[data['target'] == 1]
# print(data.count())
# # keyword       3229
# # location      2196
# # country       1578
# # text          3271
# # target        3271

# # I only need columns "keyword" and "country"
# data = data[["keyword", "country"]]
# print(data)
# print(data.count())
# # [3271 rows x 2 columns]
# # keyword    3229
# # country    1578
# print(data.describe())
# #            keyword country
# # count         3229    1578
# # unique         220      98
# # top     derailment     USA
# # freq            39     574

# # I can only use rows where country is not null
# data = data[data["country"].notnull()]
# print(data)
# print(data.describe())
# print(data.head())
# # We have a dataframe with 1578 rows

# # Now I want to know how many times each uniqe (there's 98) value of 'country' appears to get total disasters
# print(data['country'].value_counts())
# # USA    574
# # AUS    148
# # GBR    118
# # CAN    106
# # IND     96
# #       ...
# # JEY      1
# # LBY      1
# # BVT      1
# # CYP      1
# # LBR      1

# # That could be used for the map
# # totals = data['country'].value_counts()
# # print(totals.describe())
# # totals.rename(columns = {'':'country', 'country':'count'}, inplace = True)
# # print(totals.head())
# # totals.to_csv("data/totals.csv")


# PART 2: Getting disaster types

# Subset only target = 1 (Only disasters)
data = data[data['target'] == 1]

# I only need columns "keyword" and "country"
data = data[["keyword", "country"]]
# print(data.describe())

# I can only use rows where country is not null
data = data[data["country"].notnull()]
print(data.describe())
print(data.head())

# I need a new row "disaster" where values of 'keyword' will be mapped to categories
# print(data["keyword"].nunique())

data["keyword"].replace({'ablaze': 'fire'})

print(data.head())
