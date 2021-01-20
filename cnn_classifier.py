from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from numpy import genfromtxt
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import History, TensorBoard, ReduceLROnPlateau
from keras.layers import TimeDistributed
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

np.random.seed(7)
X1 = genfromtxt('reduced_9.csv', delimiter=',', dtype=None, encoding="utf-8-sig")
X2 = genfromtxt('reduced_34.csv', delimiter=',', dtype=None, encoding="utf-8-sig")
X3 = genfromtxt('reduced_38.csv', delimiter=',', dtype=None, encoding="utf-8-sig")

#X = [[X1],[X2],[X3]]
y = genfromtxt('y.csv', delimiter=',', dtype=None, encoding="utf-8-sig")

#API_names = np.unique(X, return_counts=True)
#print (len(API_names))
#print (API_names)

label_encoder = LabelEncoder()
X1 = label_encoder.fit_transform(X1)
X2 = label_encoder.fit_transform(X2)
X3 = label_encoder.fit_transform(X3)
y = label_encoder.fit_transform(y)
#print (np.unique(X))
#print (np.unique(y))
#print (X)
#print (y)
X1 = X1.reshape(len(X1), 1)
X2 = X2.reshape(len(X2), 1)
X3 = X3.reshape(len(X3), 1)
y = y.reshape(len(y), 1)
onehot_encoder = OneHotEncoder(sparse = False, categories='auto')
X1 = onehot_encoder.fit_transform(X1)
X2 = onehot_encoder.fit_transform(X2)
X3 = onehot_encoder.fit_transform(X3)
y = onehot_encoder.fit_transform(y)

X = np.column_stack([X1,X2,X3])

#print (np.shape(X))
#print (np.shape(y))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 60/30 train/test split
train_size = int(len(X) * 0.6)
train_X1 = X1[0:train_size]
train_X2 = X2[0:train_size]
train_X3 = X3[0:train_size]
test_X1 = X1[train_size:len(X)]
test_X2 = X2[train_size:len(X)]
test_X3 = X3[train_size:len(X)]

#print (type(X_train))
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
train_X1 = np.reshape(train_X1, (train_X1.shape[0], train_X1.shape[1], 1))
train_X2 = np.reshape(train_X2, (train_X2.shape[0], train_X2.shape[1], 1))
train_X3 = np.reshape(train_X3, (train_X3.shape[0], train_X3.shape[1], 1))
test_X1 = np.reshape(test_X1, (test_X1.shape[0], test_X1.shape[1], 1))
test_X2 = np.reshape(test_X2, (test_X2.shape[0], test_X2.shape[1], 1))
test_X3 = np.reshape(test_X3, (test_X3.shape[0], test_X3.shape[1], 1))

X_train = np.column_stack([train_X1,train_X2,train_X3])
X_test = np.column_stack([test_X1,test_X2,test_X3])


y_train = y
y_test = y

y_train = pad_sequences(y_train, maxlen=600)
y_test = pad_sequences(y_test, maxlen=400)

y_train = np.reshape(y_train, (X_train.shape[0], 1))
y_test = np.reshape(y_test, (X_test.shape[0], 1))
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)


#print (np.shape(X_train))
#print (np.shape(X_test))
#print (np.shape(y_train))
#print (np.shape(y_test))

input_units1 = np.shape(X_train)[1]
input_units2 = np.shape(X_train)[2]
#output_units = np.shape(y_train)[1]


model = Sequential ()
model.add(Conv1D(64, 3, activation='relu', input_shape=(input_units1, input_units2)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
#model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=100, min_lr=0.00001)
history = model.fit(X_train, y_train, batch_size= 256, verbose=1, epochs=10, validation_data=(X_test, y_test),  callbacks=[tensor_board, reduce_lr])
model.save('CNN.h5')



y_pred = model.predict(X_test, batch_size=512, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
y_test_bool = np.argmax(y_test, axis=1)

print(classification_report(y_test_bool, y_pred_bool))

cm = confusion_matrix(y_test_bool, y_pred_bool)
print("confusion matrix:")
print(cm)