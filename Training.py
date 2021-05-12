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

train_test = sys.argv[1]
train_data = pd.read_csv(train_test, sep='\t', encoding='utf-8')

# Input Data - X
x1 = list(train_data['Lane'])
x2 = list(train_data['Tile'])
x3 = list(train_data['X Coord'])
x4 = list(train_data['Y Coord'])
x5 = list(train_data['q1'])
x6 = list(train_data['q2'])
x7 = list(train_data['q3'])
x8 = list(train_data['q4'])
x9 = list(train_data['q5'])
x10 = list(train_data['q6'])
x11 = list(train_data['q7'])
x12 = list(train_data['q8'])
x13 = list(train_data['q9'])
x14 = list(train_data['q10'])

X = np.column_stack((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14))
X = np.array(X).reshape(X.shape[0], 1, X.shape[1])

# Output Target - Y
Y = np.array(list(train_data['Y_Output']))

# Train the Model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(1, 14)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

dir_path = os.path.dirname(os.path.realpath(__file__))

newpath = dir_path + "/model_checkpoints/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

filepath="model_checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(X, Y, epochs=50, validation_split=0.2, verbose=1, callbacks=callbacks_list)
