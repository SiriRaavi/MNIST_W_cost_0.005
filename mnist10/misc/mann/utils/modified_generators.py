import theano
import numpy as np
import os
from random import shuffle

from images import modified_get_shuffled_images, time_offset_input, load_transform


class modified_Generator(object):
    """docstring for Generator"""

    def __init__(self, data_folder_train, data_folder_valid, batch_size=1, num_feed_train=1, nb_classes=5,
                 nb_samples_per_class=10, max_rotation=-np.pi / 6, max_shift=0, var=0, amount=0.0, img_size=(20, 20),
                 max_iter=None):
        super(modified_Generator, self).__init__()
        self.data_folder_train = data_folder_train
        self.data_folder_valid = data_folder_valid
        self.batch_size = batch_size
        self.num_feed_train = num_feed_train
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_rotation = max_rotation * 180. / np.pi
        self.max_shift = max_shift
        self.img_size = img_size
        self.max_iter = max_iter
        self.num_iter = 0
        self.var = var
        self.amount = amount
        self.character_folders_train = [os.path.join(self.data_folder_train, family)
                                        for family in os.listdir(self.data_folder_train)
                                        if os.path.isdir(os.path.join(self.data_folder_train, family))]
        self.character_folders_valid = [os.path.join(self.data_folder_valid, family)
                                        for family in os.listdir(self.data_folder_valid)
                                        if os.path.isdir(os.path.join(self.data_folder_valid, family))]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1), self.sample(self.nb_classes)
        else:
            raise StopIteration()

    def sample(self, nb_classes):
        example_inputs = np.zeros((self.batch_size, nb_classes * self.nb_samples_per_class,
                                   np.prod(self.img_size)), dtype=theano.config.floatX)
        example_outputs = np.zeros((self.batch_size, nb_classes * self.nb_samples_per_class), dtype=np.int32)
        image_names = np.empty((self.batch_size, nb_classes * self.nb_samples_per_class), dtype=object)
        for i in range(self.batch_size):
            labels_and_images_train = modified_get_shuffled_images(self.character_folders_train, range(nb_classes),
                                                                   nb_samples=self.num_feed_train)
            shuffle(labels_and_images_train)
            labels_and_images_valid = modified_get_shuffled_images(
                self.character_folders_valid,
                range(nb_classes),
                nb_samples=self.nb_samples_per_class - self.num_feed_train)
            shuffle(labels_and_images_valid)
            labels_and_images = labels_and_images_train + labels_and_images_valid

            sequence_length = len(labels_and_images)
            labels, image_files = zip(*labels_and_images)

            angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=sequence_length)
            shifts = np.random.randint(-self.max_shift, self.max_shift + 1, size=(sequence_length, 2))

            example_inputs[i] = np.asarray([load_transform(filename, angle=angle, s=shift,
                                                           var=self.var, amount=self.amount).flatten()
                                            for (filename, angle, shift) in
                                            zip(image_files, angles, shifts)], dtype=theano.config.floatX)
            example_outputs[i] = np.asarray(labels, dtype=np.int32)
            image_names[i] = np.array([os.path.splitext(os.path.basename(image_file))[0] for image_file in image_files])
        return example_inputs, example_outputs, image_names
