import keras
import cv2
import numpy as np
from tqdm import tqdm
import pickle as pkl


def dump(path, trainx, trainy, testx, testy):
    with open(path, 'wb') as ofs:
        pkl.dump({'trainx':trainx, 'trainy':np.reshape(trainy, (-1,)), 'testx': testx, 'testy': np.reshape(testy, (-1,))}, ofs)    


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

model = keras.applications.resnet.ResNet152(include_top=False, weights='imagenet', classes=1000, pooling='avg')

x_train_fp16 = np.empty((len(x_train), 2048), np.float16)
x_test_fp16 = np.empty((len(x_test), 2048), np.float16)

iter = 0
for i in tqdm(range(int(np.ceil(len(x_train)/32))), ncols=60):
    xxx = []
    siter = iter
    for j in range(32):
        if iter>len(x_train)-1: break
        xxx.append(cv2.resize(x_train[iter], (224, 224), interpolation=cv2.INTER_LANCZOS4))
        iter+=1
    xxx = np.array(xxx)
    preprocessed_xxx = keras.applications.resnet.preprocess_input(xxx)
    f = model.predict(preprocessed_xxx)
    x_train_fp16[siter:siter+32] = f.astype(np.float16)

iter = 0
for i in  tqdm(range(int(np.ceil(len(x_test)/32))), ncols=60):
    xxx = []
    siter = iter
    for j in range(32):
        if iter>len(x_test)-1: break
        xxx.append(cv2.resize(x_test[iter], (224, 224), interpolation=cv2.INTER_LANCZOS4))
        iter+=1
    xxx = np.array(xxx)
    preprocessed_xxx = keras.applications.resnet.preprocess_input(xxx)
    f = model.predict(preprocessed_xxx)
    x_test_fp16[siter:siter+32] = f.astype(np.float16)


dump('data/cifar10_transfer.pkl', x_train_fp16, y_train, x_test_fp16, y_test)