import file_management as fm
import random
import os
import shutil
import pandas as pd
src_dir = '/home/cougarnet.uh.edu/sraavi/PycharmProjects/MNIST/mnist/train_data/'
dst_dir_itm = '.data/mnist_10/train_data/'
dst_dir = './data/mnist_10_as_2/train_data/'
src_val_dir = '/home/cougarnet.uh.edu/sraavi/PycharmProjects/MNIST/mnist/valid_data/'
dst_val_dir = './data/mnist_10_as_2/valid_data/'
log_at = './mnist10/logging/'

NOISE_START = 0
NOISE_END = 1000
TRUE_DATA_START = 1001
TRUE_DATA_END = 5000
TRUE_DATA_SAMPLES_COUNT_L1= 500
TRUE_DATA_SAMPLES_COUNT_L2= 300
NOISE_PERCENTAGE_L1 = 30
NOISE_PERCENTAGE_L2 = 10
VALID_PERCENTAGE = 50
# CAREFUL with below params
REDUCE = 50
BOOST = 20

NOISE_SAMPLES_COUNT_L1 = int(TRUE_DATA_SAMPLES_COUNT_L1 * NOISE_PERCENTAGE_L1/100)
NOISE_SAMPLES_COUNT_L2 = int(TRUE_DATA_SAMPLES_COUNT_L2 * NOISE_PERCENTAGE_L2/100)
VAL_SAMPLE_COUNT = int(TRUE_DATA_SAMPLES_COUNT_L1 * VALID_PERCENTAGE/100)



def runOnce():
    # preparing 10 classes from 10 classes
    ten_items_arr = fm.SelectItemsFromDir(src_dir,10)
    for e in ten_items_arr:
        fm.MakeDir(dst_dir_itm+e)

    #copy a sample amount to mnist 10 with 300+noise items each
    class_paths = [src_dir + cls for cls in ten_items_arr]
    items_class = [os.listdir(path) for path in class_paths]
    lol_item_seperation= [fm.MultipleRangeSelector(items, [NOISE_START, TRUE_DATA_START], [NOISE_END, TRUE_DATA_END])for items in items_class]
    train = [each[0] for each in lol_item_seperation]
    noise = [each[1] for each in lol_item_seperation]
    sample_train = [random.sample(train_each, TRUE_DATA_SAMPLES_COUNT_L1) for train_each in train]
    sample_noise = [random.sample(noise_each, NOISE_SAMPLES_COUNT_L1) for noise_each in noise]
    sample_train_paths = [fm.PathMaker(src_dir,each) for each in sample_train]
    sample_noise_paths = [fm.PathMaker(src_dir, each) for each in sample_noise]

    for i in range(len(sample_train_paths)):
        train_item = sample_train_paths[i]
        noise_items = sample_noise_paths[i]
        dst = dst_dir_itm + fm.PeekForClass(sample_train[i])
        fm.CopyAllToOne(train_item,dst)
        fm.CopyAllToOne(noise_items,dst)

def runEveryTrainIter(i):
    log = fm.Logger(log_at+"episode_"+str(i))
    log.MakeTruncatefile()
    # preparing 2 classes from 10 classes
    ten_items_arr = fm.SelectItemsFromDir(dst_dir_itm, 10)

    class_0 = ten_items_arr[0]
    class_0_path = dst_dir_itm + class_0
    items_class_0 = os.listdir(class_0_path)
    lol_item_seperation_0 = fm.MultipleRangeSelector(items_class_0,[NOISE_START,TRUE_DATA_START],[NOISE_END,TRUE_DATA_END])
    train_0 = lol_item_seperation_0[0]
    noise_0 = lol_item_seperation_0[1]
    sample_train_0 = random.sample(train_0,TRUE_DATA_SAMPLES_COUNT_L2)
    sample_noise_0 = random.sample(noise_0, NOISE_SAMPLES_COUNT_L2*(BOOST+100)/100)
    sample_train_0_paths = fm.PathMaker(dst_dir_itm,sample_train_0)
    sample_noise_0_paths = fm.PathMaker(dst_dir_itm,sample_noise_0)

    class_1 = ten_items_arr[1:]
    class_1_paths = [dst_dir_itm + cls for cls in class_1]
    items_class_1 = [os.listdir(path) for path in class_1_paths ]
    lol_item_seperation_1 = [fm.MultipleRangeSelector(items,[NOISE_START,TRUE_DATA_START],[NOISE_END,TRUE_DATA_END]) for items in items_class_1]
    train_1 = [each[0] for each in lol_item_seperation_1]
    noise_1 = [each[1] for each in lol_item_seperation_1]
    sample_train_1 = [random.sample(train_each,TRUE_DATA_SAMPLES_COUNT_L2) for train_each in train_1]
    sample_noise_1 = [random.sample(noise_each,int( NOISE_SAMPLES_COUNT_L2*REDUCE/100)) for noise_each in noise_1]
    sample_train_1_paths = [fm.PathMaker(dst_dir_itm, each) for each in sample_train_1]
    sample_noise_1_paths = [fm.PathMaker(dst_dir_itm, each) for each in sample_noise_1]

    dst_dir0 = dst_dir+"0"
    dst_dir1 = dst_dir+"1"
    fm.MakeDir(dst_dir0 )
    fm.MakeDir(dst_dir1)

    fm.CopyAllToOne(sample_train_0_paths,dst_dir0,log,"train0")
    fm.CopyAllToOne(sample_noise_0_paths,dst_dir1,log,"noise0")

    for i in range(len(sample_train_1_paths)):
        fm.CopyAllToOne(sample_train_1_paths[i],dst_dir1,log,"train1")
        fm.CopyAllToOne(sample_noise_1_paths[i],dst_dir0,log,"noise1")


def runSimpleValidCopy():
    ten_items_arr = fm.SelectItemsFromDir(src_val_dir, 10)

    class_1 = ten_items_arr
    class_1_paths = [ src_val_dir + cls for cls in class_1]
    items_class_1 = [os.listdir(path) for path in class_1_paths]
    item_list_1 = [random.sample(items,VAL_SAMPLE_COUNT)  for items in items_class_1]
    sample_train_1_paths = [fm.PathMaker(src_val_dir, each) for each in item_list_1]
    for e in class_1:
        dst_dir0 = dst_val_dir + e
        fm.MakeDir(dst_dir0)
    for each in range(len(sample_train_1_paths)):
        fm.CopyAllToOne(sample_train_1_paths[each], dst_val_dir+class_1[each])


def runLoggedValid(e_no):
    x = pd.DataFrame.from_csv(log_at+"episode_"+str(e_no),header=None,index_col=None)
    x.columns = ["label","file","class"]
    class_0_table = x.loc[x['label']=='train0']
    class_1_table = x.loc[x['label'] == 'noise1']
    class_0 = list(set(class_0_table['class']))
    class_1 = list(set(class_1_table['class']))
    class_1_paths = [src_val_dir + str(cls) for cls in class_1]
    class_0_paths = [src_val_dir + str(cls) for cls in class_0]
    items_class_1 = [os.listdir(path) for path in class_1_paths]
    items_class_0 = [os.listdir(path) for path in class_0_paths]
    item_list_1 = [random.sample(items,VAL_SAMPLE_COUNT) for items in items_class_1]
    item_list_0 = [random.sample(items,VAL_SAMPLE_COUNT) for items in items_class_0]
    sample_train_1_paths = [fm.PathMaker(src_val_dir, each) for each in item_list_1]
    sample_train_0_paths = [fm.PathMaker(src_val_dir, each) for each in item_list_0]

    dst_dir0 = dst_val_dir + "0"
    dst_dir1 = dst_val_dir + "1"
    fm.MakeDir(dst_dir0)
    fm.MakeDir(dst_dir1)
    for each in sample_train_0_paths:
        fm.CopyAllToOne(each,dst_dir0)
    for each in sample_train_1_paths:
        fm.CopyAllToOne(each,dst_dir1)


# runOnce()
# runEveryTrainIter(1)
# runSimpleValidCopy()
# runLoggedValid(100)


