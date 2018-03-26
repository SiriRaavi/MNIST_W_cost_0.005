import cPickle
import theano
import theano.tensor as T
import numpy as np
from mann.utils.modified_metrics import accuracy_instance
from mann.model import memory_augmented_neural_network
from mann.utils.modified_generators import modified_Generator
import h5py
from py_utils import count_mislabel, prepare_img_names_to_save, update_dict, prepare_dict_to_save
import os

# path to the best trained model
# model_train = './saved_models/mnist_multi_class_changed/train/model_500.save'
train_data_dir = './data/mnist_multi_class_changed/train_data'
valid_data_dir = './data/mnist_multi_class_changed/valid_data'
# path to save the results
test_path = './saved_models/mnist_multi_class_changed/test'
if not os.path.exists(test_path):
    os.makedirs(test_path)
DISPLAY_FREQ = 100
MODEL_FREQ = 500
nb_classes, nb_samples_per_class, img_size = 2, 10, (28, 28)
memory_shape, controller_size, nb_reads = (128, 40), 200, 4


def load_model(model_selected):
    input_var = T.tensor3('input')
    target_var, mislabel = T.imatrices('target', 'mis_label')

    generator = modified_Generator(data_folder_train=train_data_dir,
                                   data_folder_valid=valid_data_dir,
                                   batch_size=1,
                                   num_feed_train=1,    # number of training samples from each class as baseline
                                   nb_classes=nb_classes, nb_samples_per_class=nb_samples_per_class,
                                   img_size=img_size,
                                   max_rotation=0, max_shift=0, var=0, amount=0,
                                   max_iter=None)

    # Model
    output_var, output_var_flatten, params1 = memory_augmented_neural_network(input_var, target_var,
                                                                              batch_size=generator.batch_size,
                                                                              nb_class=generator.nb_classes,
                                                                              memory_shape=memory_shape,
                                                                              controller_size=controller_size,
                                                                              input_size=img_size[0] * img_size[1],
                                                                              nb_reads=nb_reads)
    accuracies = accuracy_instance(T.argmax(output_var, axis=2), target_var, mislabel,
                                   nb_classes=generator.nb_classes,
                                   nb_samples_per_class=generator.nb_samples_per_class,
                                   batch_size=generator.batch_size)
    cost = T.mean(T.nnet.categorical_crossentropy(output_var_flatten, target_var.flatten()))

    posterior_fn = theano.function([input_var, target_var], output_var)
    accuracy_fn = theano.function([input_var, target_var, mislabel], accuracies)
    cost_fn = theano.function([input_var, target_var], cost)

    # load the best trained model
    f = open(model_selected, 'rb')
    loaded_params = cPickle.load(f)
    f.close()

    # set the parameters
    for i in range(len(loaded_params)):
        params1[i].set_value(loaded_params[i])

    d = dict([])
    all_acc = all_mis = np.zeros((0, generator.nb_samples_per_class))
    all_loss, accs = [], np.zeros(generator.nb_samples_per_class)
    all_names = all_classes = np.zeros((0, generator.nb_samples_per_class * generator.nb_classes), dtype=np.int32)
    for i, (test_input, test_target, image_names) in generator:
        test_output = np.argmax(posterior_fn(test_input, test_target), axis=2)
        test_mislabel = count_mislabel(generator, image_names)
        acc1, _, mis = accuracy_fn(test_input, test_target, test_mislabel)
        d = update_dict(d, acc1, image_names, generator.num_feed_train)
        all_acc = np.concatenate((all_acc, acc1.reshape([-1, generator.nb_samples_per_class])))
        all_mis = np.concatenate((all_mis, mis.reshape([-1, generator.nb_samples_per_class])))
        cls_label, image_name = prepare_img_names_to_save(image_names)

        all_names = np.concatenate((all_names, image_name.reshape(
            -1, generator.nb_samples_per_class * generator.nb_classes)), axis=0)
        all_classes = np.concatenate((all_classes, cls_label.reshape(
            -1, generator.nb_samples_per_class * generator.nb_classes)), axis=0)

        loss = cost_fn(test_input, test_target)
        all_loss.append(loss)
        accs += acc1
        if i > 0 and not (i % DISPLAY_FREQ):
            print('Episode %05d: %.6f' % (i, loss))
            print(accs / 100.)
            accs = np.zeros(generator.nb_samples_per_class)
            # save the model parameters, loss and accuracy values every 500 episodes
            if i > 0 and not (i % MODEL_FREQ):
                mislabel_count = prepare_dict_to_save(d)
                h5f = h5py.File(test_path+'/test_Results.h5', 'w')
                h5f.create_dataset('all_acc_episode', data=all_acc)
                h5f.create_dataset('loss_episode', data=all_loss)
                h5f.create_dataset('names_episode', data=all_names)
                h5f.create_dataset('classes_episode', data=all_classes)
                h5f.create_dataset('mis_episode', data=all_mis)
                h5f.create_dataset('mislabel_count', data=mislabel_count)
                h5f.close()
                print('****************************************************************************************')


if __name__ == '__main__':
    model_path = './saved_models/mnist_multi_class_changed/train/'
    #print len(os.listdir(model_path))
    for i in os.listdir(model_path):
         model_selected = model_path + i
         #print model_selected
         load_model(model_selected)
