import sys
import os
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint

FQ_file = sys.argv[1]
mylines = []
with open(FQ_file) as f:
    for myline in f:  
        m = myline.replace("\n", "")
        mylines.append(m.replace("/1", ""))  
print ("***************  Info 1: Read the FASTQ File. *************** ")
names = mylines[::4]
qual_score = mylines[3::4]
print ("***************  Info 2: Extracted names and quality score. *************** ")
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

print ("***************  Info 3: Calculated quality score to numbers. *************** ")
def create_model():
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(1, 14), unroll=True))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

test_file = sys.argv[2]
test_data = pd.read_csv(test_file, sep='\t', encoding='utf-8')
test_data['Combined'] = test_data[['Lane', 'Tile', 'X Coord', 'Y Coord', 'q1', 'q2', 'q3', 'q4','q5', 'q6', 'q7', 'q8', 'q9', 'q10']].values.tolist()
test_data_fin = test_data[['Combined', 'Y_Output']]

print ("***************  Info 4: Loaded CSV File as Test Data *************** ")

model = create_model()   
model.load_weights(sys.argv[3])
model.compile(optimizer='adam', loss='mse')
min_max_scaler = MinMaxScaler()
y_init = []
for i in fin_qual:
    y_init.append((i[0:10]))

q1 = []
for index, row in test_data_fin.iterrows():
    q1.append(row['Combined'])

c = 0
t2 = []
for i in range(len(qual_score)):
    t2.append(q1[c:c+98])
    c += 98
print ("***************  Info 5: Prediction Started. *************** ")
fin_quality_list = []
for i in t2:
    y_int = []
    for j in i:
        test_output = model.predict(np.array(j).reshape((1, 1, 14)), verbose=0)
        y_int.append(list(test_output.reshape(-1,))[0])
    fin_quality_list.append(y_int)
print ("***************  Info 6: Prediction Finished *************** ")
def norm_qual(q):
    q_scaled = min_max_scaler.fit_transform(np.array(q).reshape(-1, 1))
    return list(q_scaled.reshape(-1,))
fin_norm_qual = []
for i in fin_qual:
    fin_norm_qual.append(norm_qual(i))
t = []
for i in fin_quality_list:
    t_int = min_max_scaler.inverse_transform(np.array(i).reshape(-1, 1))
    t.append(list(t_int.reshape(-1,)))
print ("***************  Info 7: Inverse Transformed the Predicted Scores. *************** ")
pred_quality_list = []
for y, i in zip(y_init,t):
    for j in i:
        y.append(j)
    pred_quality_list.append(y)

def int_to_char(l1, l2):
    if (l1 == l2) or (l1 == l2+1) or (l1 == l2-1) :
        return '0'
    else:
        return chr(l2 + 33)

print ("***************  Info 7: Comparison against original Quality Scores started. *************** ")
qual_char_pred = []
for i,j in zip(pred_quality_list, fin_qual):
    qual_str = ''
    for k,l in zip(i, j):
        qual_str += int_to_char(round(k),l)
    qual_char_pred.append(qual_str)

print ("***************  Info 8: Final Quality Score in ASCII format. *************** ")

dir_path = os.path.dirname(os.path.realpath(__file__))

newpath = dir_path + "/output/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

with open('output/predicted_qual_score.txt', 'w') as f:
    for i in qual_char_pred:
        f.write(i + "\n") 
print ("***************  Info 9: Quality Scores written to predicted_qual_score.txt . *************** ")
