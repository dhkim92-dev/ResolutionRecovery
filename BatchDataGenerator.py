## 2019.06.08
## 참고
## http://www.kwangsiklee.com/2018/11/keras%EC%97%90%EC%84%9C-sequence%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8C%80%EC%9A%A9%EB%9F%89-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EC%B2%98%EB%A6%AC%ED%95%98%EA%B8%B0/
## https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
## keras를 이용하여 Batch 처리 대용량 데이터를 생성하는 방법

import numpy as np
from tensorflow import keras
import cv2, os

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        ## X,y 각각에 대하여 배치데이터 생성
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.dim[0] * 4, self.dim[1] * 4, self.n_channels))

        #데이터 생성
        for i, ID in enumerate(list_IDs_temp):
            # 샘플 저장
            # ID.npy
            X[i] = np.load(ID)
            sep = ID.split('/')## /에 따라 경로 분리
            sep[-2] = 'y' + sep[-2][1:]
            y_path = os.path.join(os.sep, *sep)
            y[i] = np.load(y_path)

        return X, y