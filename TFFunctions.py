from __future__ import print_function
import numpy as np
import tensorflow as tf
import operator
from six.moves import cPickle as pickle
from numpy.ma import sqrt



def make_arrays(nb_rows, img_size_x, img_size_y):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size_x, img_size_y), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def save_pickle(pickle_file, save):
    try:
        f = open(pickle_file, 'wb')
        print("save:", save)
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


def reformat(dataset, labels, image_size_x, image_size_y, num_labels):
    dataset = dataset.reshape((-1, image_size_x * image_size_y)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def generate_hidden_layer_key(layer_num):
    return 'h' + str(layer_num)


# source:  http://arxiv.org/pdf/1502.01852v1.pdf
def calculateOptimalWeightStdDev(numPreviousLayerParams):
    return sqrt(2.0 / numPreviousLayerParams)


def validate_num_hidden_layers(num_hidden_layers):
    if num_hidden_layers < 1:
        raise ValueError('Number of hidden layers must be >= 1')


def generate_weights(hidden_layers, num_features, num_labels):
    num_hidden_layers = hidden_layers.__len__()
    validate_num_hidden_layers(num_hidden_layers)
    weights = {}

    num_hidden_features = hidden_layers[0]
    stddev = calculateOptimalWeightStdDev(num_features)
    weights[generate_hidden_layer_key(1)] = tf.Variable(tf.truncated_normal([num_features, num_hidden_features], 0, stddev))

    for layerNum in xrange(num_hidden_layers + 1):
        if layerNum > 1:
            previous_num_hidden_features = num_hidden_features
            num_hidden_features = hidden_layers[layerNum - 1]
            stddev = calculateOptimalWeightStdDev(previous_num_hidden_features)
            weights[generate_hidden_layer_key(layerNum)] = tf.Variable(
                tf.truncated_normal([previous_num_hidden_features, num_hidden_features], 0, stddev))

    stddev = calculateOptimalWeightStdDev(num_hidden_features)
    weights['out'] = tf.Variable(tf.truncated_normal([num_hidden_features, num_labels], 0, stddev))
    return weights


def generate_biases(hiddenLayers, numLabels):
    numHiddenLayers = hiddenLayers.__len__()
    validate_num_hidden_layers(numHiddenLayers)
    biases = {}

    numHiddenFeatures = hiddenLayers[0]
    biases[generate_hidden_layer_key(1)] = tf.Variable(tf.zeros([numHiddenFeatures]))

    for layerNum in xrange(numHiddenLayers + 1):
        if layerNum > 1:
            numHiddenFeatures = hiddenLayers[layerNum - 1]
            biases[generate_hidden_layer_key(layerNum)] = tf.Variable(tf.zeros([numHiddenFeatures]))

    biases['out'] = tf.Variable(tf.zeros([numLabels]))
    return biases


def generate_loss_calc(weights, biases, numHiddenLayers, trainingNetwork, trainingLabels, regularizationRate):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(trainingNetwork, trainingLabels))
    regularizers = generateRegularizers(weights, biases, numHiddenLayers)
    loss += regularizationRate * regularizers
    return loss


def generate_hidden_layer(layerNum, previousLayer, weights, biases, training, dropoutKeepRate):
    key = generate_hidden_layer_key(layerNum)
    if training:
        hiddenLayer = tf.nn.relu(tf.matmul(previousLayer, weights[key]) + biases[key])
        hiddenLayer = tf.nn.dropout(hiddenLayer, dropoutKeepRate)
        return hiddenLayer
    else:
        hiddenLayer = tf.nn.relu(tf.matmul(previousLayer, weights[key]) + biases[key])
        return hiddenLayer


def generateRegularizers(weights, biases, numHiddenLayers):
    validate_num_hidden_layers(numHiddenLayers)
    regularizers = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['h1'])

    for layerNum in xrange(numHiddenLayers + 1):
        if layerNum > 1:
            regularizers = regularizers + tf.nn.l2_loss(weights['h' + str(layerNum)]) + tf.nn.l2_loss(
                biases['h' + str(layerNum)])

    regularizers = regularizers + tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out'])
    return regularizers


def multilayer_network(inputs, weights, biases, numHiddenLayers, training, dropoutKeepRate):
    validate_num_hidden_layers(numHiddenLayers)

    hiddenLayer = generate_hidden_layer(1, inputs, weights, biases, training, dropoutKeepRate)

    for layerNum in xrange(numHiddenLayers + 1):
        if layerNum > 1:
            hiddenLayer = generate_hidden_layer(layerNum, hiddenLayer, weights, biases, training, dropoutKeepRate)

    return tf.matmul(hiddenLayer, weights['out']) + biases['out']


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def get_max(l):
    max_idx, max_val = max(enumerate(l), key=operator.itemgetter(1))
    return max_idx, max_val
