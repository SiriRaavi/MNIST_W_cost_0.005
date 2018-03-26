import theano
import theano.tensor as T
import numpy as np


# predictions is the argmax of the posterior
def accuracy_instance(predictions, targets, mislabel, nb_classes=2, nb_samples_per_class=10, batch_size=1):
    accuracy_0 = theano.shared(np.zeros((batch_size, nb_samples_per_class), dtype=theano.config.floatX))
    mis_0 = theano.shared(np.zeros((batch_size, nb_samples_per_class), dtype=np.int32))
    indices_0 = theano.shared(np.zeros((batch_size, nb_classes), dtype=np.int32))
    batch_range = T.arange(batch_size)

    def step_(p, t, m, acc, idx, miss):
        acc = T.inc_subtensor(acc[batch_range, idx[batch_range, t]], T.eq(p, t))
        miss = T.inc_subtensor(miss[batch_range, idx[batch_range, t]], m)
        idx = T.inc_subtensor(idx[batch_range, t], 1)
        return acc, idx, miss

    (raw_accuracy, _, raw_mislabel), _ = theano.foldl(step_, sequences=[predictions.dimshuffle(1, 0),
                                                                        targets.dimshuffle(1, 0),
                                                                        mislabel.dimshuffle(1, 0)],
                                                      outputs_info=[accuracy_0, indices_0, mis_0])
    accuracy = T.mean(raw_accuracy / nb_classes, axis=0)
    mis_label = T.sum(raw_mislabel, axis=0)

    return accuracy, raw_accuracy, mis_label
