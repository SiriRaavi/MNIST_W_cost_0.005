import os
import random
import shutil

def class_selection(data_dir):
    class0 = random.choice(os.listdir(data_dir))
    class1 = [x for x in os.listdir(data_dir) if x is not class0]
    return class0, class1

def copy_files(src, dst):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dst)

def copy_dir():
    data_dir = './data/mnist_10/train_data/'
    y1 = class_selection(data_dir)[0]

    dst_dir_1 = './data/mnist_10_as_2/train_data/0/'
    try:
        shutil.rmtree(dst_dir_1) #removes it for every iteration
        os.makedirs(dst_dir_1)
    except Exception, e:
        print e
        pass

    dst_dir_2 = './data/mnist_10_as_2/train_data/1/'
    try:
        shutil.rmtree(dst_dir_2)
        os.makedirs(dst_dir_2)
    except Exception, e:
        print e
        pass

    src_dir_1 = data_dir + y1 + '/'
    print 'Random selection:', y1

    data_dir_list = os.listdir(data_dir)

    for i in data_dir_list:
        print i
        if i == y1:
            for j in os.listdir(data_dir + '/' + i):
                f_name = j
                if int(j.replace('.png', '')[2:]) >= 1000:
                    #print data_dir + '/' + i + '/' + f_name
                    shutil.copy(data_dir + '/' + i + '/' + f_name, dst_dir_1)
                else:
                    shutil.copy(data_dir + '/' + i + '/' + f_name, dst_dir_2)

        else:
            for j in os.listdir(data_dir + '/' + i):
                f_name = j
                if int(j.replace('.png', '')[2:]) >= 1000:
                    #print data_dir + '/' + i + '/' + f_name
                    shutil.copy(data_dir + '/' + i + '/' + f_name, dst_dir_2)
                else:
                    shutil.copy(data_dir + '/' + i + '/' + f_name, dst_dir_1)

copy_dir()