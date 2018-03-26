from __future__ import print_function

import cPickle
import theano
import theano.tensor as T
import numpy as np
import h5py
import lasagne.updates
from mann.utils.generators import Generator
from mann.utils.metrics import accuracy_instance
from mann.model import memory_augmented_neural_network
import time
import os
import class_selection as cs


DISPLAY_FREQ = 100
MODEL_FREQ = 500
batch_size, nb_classes, nb_samples_per_class, img_size = 16, 2, 10, (28, 28)
memory_shape, controller_size, nb_reads = (128, 40), 200, 4
data_dir = './data/mnist_multi_class_changed/train_data'
save_dir = './saved_models/mnist_multi_class_changed'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(save_dir+'/train')
    os.makedirs(save_dir+'/valid')


def MANN_train():
    cs.copy_dir()
    input_var = T.tensor3('input')  # input_var has dimensions (batch_size, time, input_dim)
    target_var = T.imatrix('target')  # target_var has dimensions (batch_size, time) (label indices)

    # Load data
    generator = Generator(data_folder=data_dir,
                          batch_size=batch_size,
                          nb_classes=nb_classes, nb_samples_per_class=nb_samples_per_class, img_size=img_size,
                          max_rotation=np.pi / 6., max_shift=0, var=0, amount=0.05,
                          max_iter=None)

    output_var, output_var_flatten, params = memory_augmented_neural_network(input_var,
                                                                             target_var,
                                                                             batch_size=generator.batch_size,
                                                                             nb_class=generator.nb_classes,
                                                                             memory_shape=memory_shape,
                                                                             controller_size=controller_size,
                                                                             input_size=img_size[0] * img_size[1],
                                                                             nb_reads=nb_reads)

    cost = T.mean(T.nnet.categorical_crossentropy(output_var_flatten, target_var.flatten()))
    updates = lasagne.updates.adam(cost, params, learning_rate=1e-3)
    posterior_fn = theano.function([input_var, target_var], output_var)
    accuracies = accuracy_instance(T.argmax(output_var, axis=2), target_var,
                                   nb_classes=generator.nb_classes,
                                   nb_samples_per_class=generator.nb_samples_per_class,
                                   batch_size=generator.batch_size)

    print('Compiling the model...')
    train_fn = theano.function([input_var, target_var], cost, updates=updates)
    accuracy_fn = theano.function([input_var, target_var], accuracies)
    print('Done')
    print('Training...')
    t0 = time.time()
    losses, accs = [], np.zeros(generator.nb_samples_per_class)
    all_acc = np.zeros((0, generator.nb_samples_per_class))
    all_loss = np.array([])
    d = dict([])
    try:
        for i, (example_input, example_output, example_name) in generator:

            loss = train_fn(example_input, example_output)
            acc = accuracy_fn(example_input, example_output)
            pred_output = np.argmax(posterior_fn(example_input, example_output), axis=2)

            # all_loss and all_acc stores all the values up to the end of training
            # These will be saved later for the plotting and visualizing purposes
            all_acc = np.concatenate((all_acc, acc.reshape([-1, generator.nb_samples_per_class])))
            all_loss = np.append(all_loss, loss)

            # losses and accs are storing the values for displaying purpose (during the training)
            # and will be reset after each display
            losses.append(loss)
            accs += acc
            if i > 0 and not (i % DISPLAY_FREQ):
                print('Episode %05d: %.6f' % (i, np.mean(losses)))
                print(accs / 100.)
                losses, accs = [], np.zeros(generator.nb_samples_per_class)
                cs.copy_dir()


            # save the model parameters, loss and accuracy values every 5000 episodes
            if i > 0 and not (i % MODEL_FREQ):
                f = open(save_dir + '/train/model_' + str(i) + '.save', 'wb')
                saved_params = [params[i].get_value() for i in range(len(params))]
                cPickle.dump(saved_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()

                h5f = h5py.File(save_dir + '/Results.h5', 'w')
                h5f.create_dataset('all_acc', data=all_acc)
                h5f.create_dataset('loss', data=all_loss)
                h5f.close()

    except KeyboardInterrupt:
        print(time.time() - t0)
        pass


if __name__ == '__main__':
    MANN_train()
