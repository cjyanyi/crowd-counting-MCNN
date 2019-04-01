import numpy as np
import os, shutil

def isProtest_test():
    ori_data = np.genfromtxt('annot_test.txt', delimiter='\t', usecols=[1], dtype=None, encoding=None, skip_header=1)
    return np.mat(ori_data).T

def isProtest_train():
    ori_data = np.genfromtxt('annot_train.txt', delimiter='\t', usecols=[1], dtype=None, encoding=None, skip_header=1)
    return np.mat(ori_data).T

def protestInfo_test():
    ori_data = np.genfromtxt('annot_test.txt', delimiter='\t', usecols=(list(range(1, 13))), dtype=None, encoding=None, skip_header=1)
    protest_data = [[int(x) for x in list(row)[2:]] + [float(list(row)[1])] for row in ori_data if row[0]]
    return np.array(protest_data)

def protestInfo_train():
    ori_data = np.genfromtxt('annot_train.txt', delimiter='\t', usecols=(list(range(1, 13))), dtype=None, encoding=None, skip_header=1)
    protest_data = [[int(x) for x in list(row)[2:]] + [float(list(row)[1])] for row in ori_data if row[0]]
    return np.array(protest_data)

def pickProtestPic_test(path):
    ori_data = np.genfromtxt(path+'annot_test.txt', delimiter='\t', usecols=[0, 1], dtype=None, encoding=None, skip_header=1)
    protest_pic = [row for row in ori_data if row[1]]

    os.makedirs(path+"protest_test")
    for pic in protest_pic:
        shutil.copyfile(path+"test/"+pic[0], path+"protest_test/"+pic[0])

def pickProtestPic_train(path):
    ori_data = np.genfromtxt(path+'annot_train.txt', delimiter='\t', usecols=[0, 1], dtype=None, encoding=None, skip_header=1)
    protest_pic = [row for row in ori_data if row[1]]

    os.makedirs(path+"protest_train")
    for pic in protest_pic:
        shutil.copyfile(path+"train/"+pic[0], path+"protest_train/"+pic[0])

def pickProtestPic(fname):
    ori_data = np.genfromtxt(fname, delimiter='\t', usecols=[0, 1], dtype=None, encoding=None,
                             skip_header=1)
    protest_pic = [row for row in ori_data if row[1]]

    os.makedirs("protest_train")
    for pic in protest_pic:
        shutil.copyfile("train/" + pic[0], "protest_train/" + pic[0])

def protestInfo(fname):
    ori_data = np.genfromtxt(fname, delimiter='\t', usecols=(list(range(1, 13))),
                             dtype=None, encoding=None, skip_header=1)
    protest_data = [[int(x) for x in list(row)[2:]] + [float(list(row)[1])] for row in ori_data if row[0]]
    return np.array(protest_data)
    #protest_labels, violence = [[int(x) for x in list(row)[2:]] for row in ori_data if row[0]], [float(list(row)[1]) for row in ori_data if row[0]]
    #return np.array(protest_labels), np.array(violence)


