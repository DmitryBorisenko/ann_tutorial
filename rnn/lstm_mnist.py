"""Build an LSTM network for the MNIST task in TensorFlow. Treating each
28 x 28 image as 28 sequences of length 28.

"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_and_prepare_data():
    """Utility function fetching the default MNIST dataset and reshaping the
    images from (784,) vectors into (28, 28) matrices.

    Returns
    -------
    train_X: np.array
        of shape (55000, 28, 28) with MNIST images for training
    train_y: np.array
        of shape (55000, 10) with one-hot encoded labels for training
    test_X: np.array
        of shape (10000, 28, 28) with MNIST images for testing
    test_y:
        of shape (10000, 10) with one-hot encoded labels for testing

    """
    # Load data into the current directory
    mnist_data = input_data.read_data_sets(".", one_hot=True)

    train_X = mnist_data.train.images
    train_y = mnist_data.train.labels
    test_X = mnist_data.test.images
    test_y = mnist_data.test.labels

    # Reshape the examples into images
    train_X = np.reshape(train_X, (train_X.shape[0], 28, 28))
    test_X = np.reshape(test_X, (test_X.shape[0], 28, 28))

    return train_X, train_y, test_X, test_y


def main(n_hidden, learning_rate, batch_size, training_iters):
    """Constructs and trains an LSTM network for the MNIST task.

    Parameters
    ----------
    n_hidden: int
        number of neurons in the hidden layer
    learning_rate: float
        learning rate for the Adam optimizer
    batch_size: int
        size of the minibatch - number of examples fed into the network per
        gradient update
    training_iters: int
        how many batches of data the network will see before stopping training

    Returns
    -------
    Nophin' trains a simple LSTM on MNIST data, printing some outputs

    """
    # Load the data
    train_X, train_y, test_X, test_y = load_and_prepare_data()

    # Construct the network ===================================================

    # Feed images into LSTM row by row. That is in LSTM terminology the first,
    # vertical dimension of an input image is the number of time steps and the
    # second, horizontal dimension is the number of sequences.
    n_steps = train_X.shape[1]  # timesteps
    n_seq = train_X.shape[2]    # MNIST data input (img shape: 28*28)
    n_classes = train_y.shape[1]

    # The input is a (?, 28, 28) tensor: (batch size, timesteps, seq length)
    x = tf.placeholder(dtype="float", shape=[None, n_steps, n_seq], name="x")

    # Construct the LSTM network
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    output, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)

    # The LSTM output has shape (batch size, timesteps, hidden layer size)
    # Get the hidden layer for the last time step - (batch size, n_hidden)
    out = tf.reshape(tf.split(output, n_steps, axis=1, name="split")[-1],
                     [-1, n_hidden])

    # Linearly map the (batch size, n_hidden) layer to labels
    weights_out = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    biases_out = tf.Variable(tf.random_normal([n_classes]))

    y_hat = tf.matmul(out, weights_out) + biases_out

    # The true label is a (batch_size, number of classes) 2-tensor
    y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y")

    # Set up the optimizer and metrics ========================================
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        )
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Train the network
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 1

        # Keep training until reach max iterations
        while step * batch_size < training_iters:

            # Randomly select a batch
            batch_idx = np.random.choice(train_X.shape[0], batch_size)
            batch_x = train_X[batch_idx, :, :]
            batch_y = train_y[batch_idx, :]

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if step % 10 == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

            step += 1

        print("Optimization finished.")

        # Calculate accuracy for 128 mnist test images
        test_len = 256
        test_data = test_X[:test_len].reshape((-1, n_steps, n_seq))
        test_label = test_y[:test_len]

        print(
            "Testing Accuracy:",
            sess.run(accuracy, feed_dict={x: test_data, y: test_label})
            )

    sess.close()


if __name__ == "__main__":
    main(n_hidden=128, learning_rate=1e-3, batch_size=32, training_iters=1e5)
