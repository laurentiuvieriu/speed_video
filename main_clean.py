from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, TimeDistributed, Input, BatchNormalization, Activation
from keras import backend as K

import csv
from utils import computeMeanImageFromFolder, generateWindowFromIdx
import numpy as np

root = 'path_to_root/'
labelFile = root + "data/ytrain.txt"
noSamp_train = 20400
batch_size = 256
window_size = 10
img_size = 32
shuff_batch_ind = np.random.permutation(range(noSamp_train - window_size))

with open(labelFile, 'rb') as df:
    csvreader = csv.reader(df, delimiter=' ')
    ylabel = []
    for row in csvreader:
        ylabel.append(float(row[0]))

ylabel = np.asarray(ylabel)
ylabel = ylabel/30.0

input_shape = (img_size, img_size, 3)
input_image = Input(shape=(img_size,img_size,3))
input_sequence = Input(shape=(window_size, img_size, img_size, 3))

mean_img = computeMeanImageFromFolder(root+"data/train_{:04d}_{:04d}/".format(img_size, img_size), '.jpg', (img_size, img_size, 3))

K.set_learning_phase(1) #set learning phase

vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
vision_model.add(BatchNormalization())
vision_model.add(Activation('relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), padding='same'))
vision_model.add(BatchNormalization())
vision_model.add(Activation('relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())
vision_model.add(Dense(512, activation='relu'))
vision_model.summary()

processed_sequences = TimeDistributed(vision_model)(input_sequence)

lstm_seq = LSTM(64)(processed_sequences)
output = Dense(1, activation='linear')(lstm_seq)

full_model = Model(inputs=input_sequence, outputs=output)
full_model.summary()

full_model.compile(loss='mean_squared_error', optimizer='adam')

mse_save = []
for epoch in range(10):
    for k in range(0,len(shuff_batch_ind)- batch_size,batch_size):
        # k = 0
        batch = []
        labels = []
        for i in range(batch_size):
            batch.append(generateWindowFromIdx(window_size, shuff_batch_ind[i+ k], root+"data/train_{:04d}_{:04d}/".format(img_size, img_size), mean_img))
            labels.append(np.mean(ylabel[shuff_batch_ind[i+ k]:shuff_batch_ind[i+ k]+ window_size]))
        
        batch = np.asarray(batch)
        labels = np.asarray(labels)
        local_mse = full_model.train_on_batch(batch, labels)
        mse_save.append(local_mse)
        print("-----> epoch: {:02d}, batch: {:04d}, mse: {:4.5f}".format(epoch+ 1, k+ 1, local_mse))


# # build training set ...
# allX = []
# allY = []
# print("--> reading data ...")
# for k in range(noSamp_train - window_size):
#     if np.mod(k, 1000) == 0:
#         print("-----> frame idx: {:04d}".format(k))
#     
#     local_batch = generateWindowFromIdx(window_size, k, root+"data/train_{:04d}_{:04d}/".format(img_size, img_size), mean_img)
#     allX.append(local_batch)
#     allY.append(np.mean(ylabel[k:k+ window_size]))
# 
# allX = np.asarray(allX)
# allY = np.asarray(allY)
#  
# full_model.fit(allX, allY, batch_size=256, epochs=50, shuffle= 1, validation_split=0.25)

