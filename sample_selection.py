import os
import random
import shutil


# TRAIN
TRAIN_START = 1000
TRAIN_END = 2999
TRAIN_QUANTITY = 500

# NOISE
NOISE_START = 0
NOISE_END = 999
NOISE_QUANTITY = 10

TRAIN_SRC = '/home/siri/PycharmProjects/MNIST/train_data/'
TRAIN_DEST = './data/mnist_10/train_data/'

# VALID
VALID_START = 0
VALID_END = 500
VALID_QUANTITY = 300

VALID_SRC = '/home/siri/PycharmProjects/MNIST/valid_data/'
VALID_DEST = './data/mnist_10/valid_data/'


def string2Number(img_name):
    img_name = str(img_name).replace('.png','')
    return int(img_name[2:])


def sample_selector(src,dest,train=False):
    for i in os.listdir(src):
        src2 = src+i

        if train:
            train_arr = []
            noise_arr = []
            # test_arr = []

            for j in os.listdir(src2):
                if string2Number(j)>TRAIN_START and string2Number(j)<TRAIN_END:
                    train_arr.append(j)
                elif string2Number(j)>NOISE_START and string2Number(j)<NOISE_END:
                    noise_arr.append(j)
                # else:
                    # test_arr.append(j)

            train_sample = random.sample(train_arr,TRAIN_QUANTITY)
            noise_sample = random.sample(noise_arr,NOISE_QUANTITY)
            # test_sample = random.sample(test_arr,TEST_QUANTITY)

            for each in train_sample:
                shutil.copy(src2+'/'+each,dest+i)
            for each in noise_sample:
                shutil.copy(src2+'/'+each,dest+i)
            # for each in test_sample:
                # shutil.copy(src2+'/'+each,TEST_DIR+i)

        else:
            valid_arr = []

            for j in os.listdir(src2):
                if string2Number(j) > VALID_START and string2Number(j) < VALID_END:
                    valid_arr.append(j)
            valid_sample = random.sample(valid_arr, VALID_QUANTITY)

            for each in valid_sample:
                shutil.copy(src2 + '/' + each, dest + i)


def noise_percentage_cal():
    noise_0 = NOISE_QUANTITY * 9
    contents_0 = TRAIN_QUANTITY + noise_0
    contents_1 = (TRAIN_QUANTITY * 9) + NOISE_QUANTITY
    noise_fraction = float(noise_0)/contents_0
    return int(((contents_1 * noise_fraction) - NOISE_QUANTITY)/(1 - noise_fraction))



# sample_selector(TRAIN_SRC,TRAIN_DEST,True)

# print noise_percentage_cal()



