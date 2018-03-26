import os
import random
import shutil


#TRAIN
TRAIN_START = 1000
TRAIN_END = 2999
TRAIN_QUANTITY = 700

#NOISE
NOISE_START = 0
NOISE_END = 999
NOISE_QUANTITY = 10

TRAIN_SRC = '/home/siri/PycharmProjects/MNIST/train_data/'
TRAIN_DEST = './data/mnist_10/train_data/'

#VALID
VALID_START = 0
VALID_END = 500
VALID_QUANTITY = 300

VALID_SRC = '/home/siri/PycharmProjects/MNIST/valid_data/'
VALID_DEST = './data/mnist_10/valid_data/'

#TEST
TEST_START = 3000
TEST_END = 5000
TEST_QUANTITY = 500


def string2Number(img_name):
    img_name = str(img_name).replace('.png','')
    return int(img_name[2:])

def sample_selector(src,dest,train=False):
    for i in os.listdir(src):
        src2 = src+i

        if train:
            train_arr = []
            noise_arr = []
            #test_arr = []

            for j in os.listdir(src2):
                if string2Number(j)>TRAIN_START and string2Number(j)<TRAIN_END:
                    train_arr.append(j)
                elif string2Number(j)>NOISE_START and string2Number(j)<NOISE_END:
                    noise_arr.append(j)
                #else:
                    #test_arr.append(j)

            train_sample = random.sample(train_arr,TRAIN_QUANTITY)
            noise_sample = random.sample(noise_arr,NOISE_QUANTITY)
            #test_sample = random.sample(test_arr,TEST_QUANTITY)

            for each in train_sample:
                shutil.copy(src2+'/'+each,dest+i)
            for each in noise_sample:
                shutil.copy(src2+'/'+each,dest+i)
            #for each in test_sample:
                #shutil.copy(src2+'/'+each,TEST_DIR+i)

        else:
            valid_arr = []

            for j in os.listdir(src2):
                if string2Number(j) > VALID_START and string2Number(j) < VALID_END:
                    valid_arr.append(j)
            valid_sample = random.sample(valid_arr, VALID_QUANTITY)

            for each in valid_sample:
                shutil.copy(src2 + '/' + each, dest + i)




sample_selector(TRAIN_SRC,TRAIN_DEST,True)



