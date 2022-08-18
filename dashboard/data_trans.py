# DATA TRANSFORMATIONS TO FEED MAP

import pandas as pd

data = pd.read_csv("data/nloc_train.csv")

print(data.count())
# keyword       7552
# location      5080
# country       3481 (So, 3481 out of 5080)
# text          7613
# target        7613

# Subset only target = 1 (Only disasters)
disasters = data[data['target'] == 1]
print(disasters.count())
# keyword       3229
# location      2196
# country       1578
# text          3271
# target        3271

# Generate total counts for each country
