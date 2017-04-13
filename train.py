import tensorflow as tf
from TFFunctions import *
pickle_save_str = "/mnt/tt2botdata/data_pickle"
num_features = 5
start_learning_rate = 0.5
learning_decay_rate = 0.98
decay_Steps = 1000
num_Steps = 100000
regularization_rate = 0.001
dropout_keep_rate = 1
hidden_layers = [200, 100, 50, 20]


def reformat_features(dataset, labels, num_features, num_labels):
    dataset = dataset.reshape((-1, num_features )).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def train_graph():
    #print("image_size ", self.size_x, " ", self.size_y, "num_labels ", self.num_classes)



    with open(pickle_save_str, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        num_product_lines = save['product_line_classes']
        num_examples = save['num_examples']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    _trainSubset = num_examples # for now allways the same.


    train_dataset, train_labels = reformat_features(train_dataset, train_labels, num_features, num_product_lines)
    valid_dataset, valid_labels = reformat_features(valid_dataset, valid_labels, num_features,  num_product_lines)
    test_dataset, test_labels = reformat_features(test_dataset, test_labels, num_features,  num_product_lines)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    batch_size = 128

    num_hidden_layers = len(hidden_layers)

    tf.reset_default_graph()

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_product_lines))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights = generate_weights(hidden_layers, num_features, num_product_lines)
        # print(weights)
        biases = generate_biases(hidden_layers, num_product_lines)
        training_network = multilayer_network(tf_train_dataset, weights, biases, num_hidden_layers
                                              , True, dropout_keep_rate)
        loss = generate_loss_calc(weights, biases, num_hidden_layers, training_network
                                  , tf_train_labels, regularization_rate)
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step
                                                   , decay_Steps, start_learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        train_prediction = tf.nn.softmax(
            multilayer_network(tf_train_dataset, weights, biases, num_hidden_layers, False, dropout_keep_rate))
        valid_prediction = tf.nn.softmax(
            multilayer_network(tf_valid_dataset, weights, biases, num_hidden_layers, False, dropout_keep_rate))
        test_prediction = tf.nn.softmax(
            multilayer_network(tf_test_dataset, weights, biases, num_hidden_layers, False, dropout_keep_rate))

        oSaver = tf.train.Saver()
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name)
            print(v.value())

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in xrange(num_Steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (np.random.randint(1, _trainSubset) * batch_size) % (train_labels.shape[0]
                                                                               - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % 500 == 0):
                print("Minibatch loss at step", step, ":", l)
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        #oSaver.save(session, self.pickle_tf_model_file)  # filename ends with .ckpt
        #session.close()


train_graph()

