# -*- coding:UTF-8 -*-
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from prepare_utl import protestInfo
import tqdm

def data_preprocess(path):
    print('loading test data from dataset', path, '...')
    img_names = os.listdir(path)
    img_num = len(img_names)

    data = []
    for img_path in img_names:
        img = image.load_img(path+img_path, target_size=(768, 1024), color_mode='grayscale')
        img = image.img_to_array(img)
        #img = np.array(img)
        img = (img - 127.5) / 128
        # print(img.shape)
        #print(img.shape)
        data.append(img)

    print('load data finished.')
    #return data
    return np.array(data)

def predict1(data):
    model = model_from_json(open('keras_modelB/model.json').read())
    model.load_weights('keras_modelB/weights.h5')

    for d in data:
        inputs = np.expand_dims(d, 0)
        outputs = model.predict(inputs)

        c_pre = np.sum(outputs)
        print('pre:', c_pre)
        print(outputs)

def predict(data):
    model = model_from_json(open('keras_modelB/model.json').read())
    model.load_weights('keras_modelB/weights.h5')

    n=len(data)
    outputs= model.predict(data)
    # sum based on row
    predictions = np.sum(outputs,axis=(1,2))
    print(predictions)
    print(predictions.shape)
    np.savetxt('results.csv',predictions,delimiter=',')
    return predictions

label = protestInfo("../news_imgs/annot_test.txt")

def cal_acc(predict, path = ''):
    label_name = ['group_20', 'group_100']

    corr = 0
    total = predict.shape[0]

    lbase = 5
    for t in range(total):
        if predict[t] > 1. and label[t][lbase+0] == 1:
            corr += 1
    print("{}: {}".format(label_name[0], corr/total))

    corr=0
    miss =  {}
    for t in range(total):
        if predict[t] > 160. and label[t][lbase+1] == 1:
            corr += 1
        elif predict[t] > 160. and label[t][lbase+1] == 0:
            print('not 100: ',t+1, predict[t])
            miss[t] = predict[t]
        elif predict[t]<160 and label[t][lbase+1] == 0:
            corr +=1
    print("{}: {}".format(label_name[1], corr/total))
    print('label width: ', label.shape[1])

    img_names = os.listdir(path)
    img_names.sort()

    miss_ids = {}
    for i,x in enumerate(img_names):
        if i in miss:
            print(x)
            miss_ids[x] = miss[i]
    print(len(miss))

    # with open('100miss_test.txt', 'w') as fp:
    #     for s in miss_ids:
    #         fp.write("%s\n" % s)

    import pandas as pd
    df = pd.DataFrame.from_dict(miss_ids, orient="index")
    df.to_csv("100miss_test_ids.csv", header=False)






if __name__ == '__main__':
    #predict(data_preprocess('../news_imgs/protest_test/'))
    # predict(data_preprocess('news/'))
    from numpy import genfromtxt
    data = genfromtxt('results0.csv', delimiter=',')
    # np.reshape(data,(len(data),1))
    print(data.shape)
    cal_acc(data,'../news_imgs/protest_test/')
    print(data.tolist())
