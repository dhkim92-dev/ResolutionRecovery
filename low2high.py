import cv2
import numpy as np
import glob, os
import tensorflow as tf
import sys
from tensorflow.keras.layers import Conv2D, Input, Activation
from tensorflow.keras.models import Model,model_from_json
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.callbacks import ModelCheckpoint
from Subpixel import Subpixel
from BatchDataGenerator import DataGenerator
from matplotlib import pyplot as plt

## 전처리된 데이터들 외장하드로 빼놨음.
#prp_file_path = '/home/elensar92/workspace/resolution/ppdone/'
prp_file_path = './image/'
#test_file_path = '/media/elensar92/NTFS_SHARING/Training/'
#x_train_data_list = sorted(glob.glob(os.path.join(prp_file_path,'x_train','*.npy'))) ##용량 적은걸로 해야할듯함
#x_valid_data_list = sorted(glob.glob(os.path.join(prp_file_path,'x_valid','*.npy'))) 
#x_train = np.load(x_train_data_list[1])
#print('shape of inputs : ',np.shape(x_train))
upscale = 4

def training(model) :
    train_datas = DataGenerator(list_IDs = x_train_data_list, labels = None, batch_size = 32, dim = (44,44), n_channels = 3, n_classes =None, shuffle = True)
    valid_datas = DataGenerator(list_IDs = x_valid_data_list, labels = None, batch_size = 32, dim = (44,44), n_channels = 3, n_classes =None, shuffle = False)

    model.compile(optimizer = 'adam', loss = 'mse') ## 이미지는 mse가 더 효율적
    history = model.fit_generator(train_datas, 
        validation_data = valid_datas,
        epochs=10, 
        verbose=1, 
        callbacks=[ModelCheckpoint("./models/test1.h5", 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True)])
    
def testing(model,idx,x_test_list,y_test_list) :
    #print("testing 진입")
    x1_test = np.load(x_test_list[idx])
    t1_test = np.load(y_test_list[idx])

    print('x1_test.shape : ',x1_test.shape)
    print('t1_test_shape : ',t1_test.shape)

    model.load_weights('./models/test1.h5')
    model.compile(loss = 'mse', optimizer='adam')

    pred_img = model.predict(x1_test.reshape((1,44,44,3)))

    x1_img = (x1_test * 255).astype(np.uint8)## 노멀라이즈 복구
    pred_img = np.clip(pred_img.reshape((176,176,3)),0,1)
    x1_img = cv2.cvtColor(x1_img,cv2.COLOR_BGR2RGB)
    pred_img = cv2.cvtColor(pred_img,cv2.COLOR_BGR2RGB)
    t1_img = (t1_test*255).astype(np.uint8)
    t1_img = cv2.cvtColor(t1_img,cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=[15,5])
    plt.subplot(1,4,1)
    plt.title('test_input')
    plt.imshow(x1_img)
    plt.subplot(1,4,2)
    plt.title('output')
    plt.imshow(pred_img)
    plt.subplot(1,4,3)
    plt.title('ground_truth')
    plt.imshow(t1_img)
    plt.show()
    
def createModel() :
    input_data = Input(shape=(44,44,3))
    x = Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu')(input_data)
    x = Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu')(x)
    x = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(x)
    x = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(x)
    x = Conv2D(filters = upscale**2, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(x)
    x = Subpixel(filters = 3, kernel_size = 3, r=upscale, padding='same')(x)
    x = Activation('relu')(x)
    model = Model(input_data, x)

    return model

if __name__ == '__main__' :
    if len(sys.argv) == 1 :
        print("테스트할 이미지의 인덱스를 입력하세요.")
        print("기본으로 포함된 이미지의 수는 14개 입니다.(idx range 0-13)")
        exit(-1)
    elif len(sys.argv) == 2 :
        img_file_name = sys.argv[1]
        img_idx = int(img_file_name)
        try :
            model = createModel()
            x_test_list = sorted(glob.glob(os.path.join(prp_file_path, 'x_test', '*.npy')))
            y_test_list = sorted(glob.glob(os.path.join(prp_file_path, 'y_test', '*.npy')))
            testing(model, img_idx, x_test_list, y_test_list)
        except :
            print('인덱스 범위 초과')
            exit(-1)
    else : 
        print("사용법 : python low2high.py (image 폴더 내 저장된 이미지 갯수 내의 인덱스)")
        exit(-1)
    """
    model = createModel()
    #training(model)
    x_test_list = sorted(glob.glob(os.path.join(prp_file_path, 'x_test', '*.npy')))
    y_test_list = sorted(glob.glob(os.path.join(prp_file_path, 'y_test', '*.npy')))
    testing(model,123, x_test_list, y_test_list)
    """