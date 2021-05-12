import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler

# Input FASTQ File
FQ_file = sys.argv[1]
mylines = []
with open(FQ_file) as f:
    for myline in f:  
        m = myline.replace("\n", "")
        mylines.append(m.replace("/1", ""))  

names = mylines[::4]
qual_score = mylines[3::4]

x_1 = []
for i in names:
    x = i.split(":")
    x_1.append(x)

lane = []
tile = []

x_coord = []
y_coord = []

for i in x_1:
    lane.append(int(i[1]))
    tile.append(int(i[2]))
    x_coord.append(int(i[3]))
    y_coord.append(int(i[4]))

data_dict = {'Lane': pd.Series(lane),
             'Tile': pd.Series(lane),
             'X Coord': pd.Series(x_coord),
             'Y Coord': pd.Series(y_coord),
            }
df = pd.DataFrame(data_dict)

def base_qual(qual):
    qual_score = []
    sum_lst = 0
    m = 0.0000
    for i in qual:
        qual_score.append((ord(i) - 33))
    return qual_score
fin_qual = []
for i in qual_score:
    fin_qual.append(base_qual(i))

# Normalization

x = df.values #returns a numpy array
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_norm = pd.DataFrame(x_scaled)

df_norm['Combined'] = df_norm[[0, 1, 2, 3]].values.tolist()
comb = list(df_norm['Combined'])

def norm_qual(q):
    q_scaled = min_max_scaler.fit_transform(np.array(q).reshape(-1, 1))
    return list(q_scaled.reshape(-1,))

# Normalize Quality scores row wise
fin_norm_qual = []
for i in fin_qual:
    fin_norm_qual.append(norm_qual(i))

data_dict = {
             'Combined_name': pd.Series(comb),
             'Quality Score': pd.Series(fin_norm_qual),
            }

df = pd.DataFrame(data_dict)

def split_qual(s):
    d = []
    for i in range(len(s)-10):
        d.append(s[i:i+10])
    return d
def make_train(l, d):
    d3 = []
    for i in d:
        t = []
        for j in l:
            t.append(j)
        for z in range(10):
            t.append(i[z])
        d3.append(t)
    return d3

qual_comb_split = []
for index, row in df.iterrows():
    temp_split_qual = split_qual(row['Quality Score'])
    qual_comb_split.append(make_train(row['Combined_name'], temp_split_qual))
  
row_train = []
for i in qual_comb_split:
    for k in i:
        row_train.append(k)

train_data = {'Combined': pd.Series(row_train),
            }
df_train = pd.DataFrame(train_data)
df_final = pd.DataFrame(df_train['Combined'].to_list(), columns=['Lane', 'Tile', 'X Coord', 'Y Coord', 'q1','q2','q3','q4','q5','q6','q7','q8','q9','q10'])

# Output Y - Target
y_train = []
for index, row in df.iterrows():
    for i in range(10, len(row['Quality Score'])):
        y_train.append(row['Quality Score'][i])
df_final['Y_Output'] = y_train

# Writing to CSV
df_final.to_csv('train_data.csv', sep='\t', encoding='utf-8')