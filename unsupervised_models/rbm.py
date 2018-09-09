import tensorflow as tf
import numpy as np
from theano_utils import tile_raster_images, scale_to_unit_interval
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib.pyplot as plt


class BinaryRBM:
    """Constructs and trains the plain vanilla restricted Boltzmann machine.
    See Salakhutdinov (2015), Annual Review of Statistics and Its Application
    for an overview. Note, although it is assumed that the values in the input
    layer are binary, this behavior is not enforced, as the binary RBMs can
    handle real-valued inputs on the interval [0, 1] reasonably well.

    """

    def __init__(self, n_visible, n_hidden, learning_rate=1.0,
                 initial_weights=None):
        """Instantiates the class, randomly initializing weights and biases if
        not provided.

        Parameters
        ----------
        n_visible: int
            number of units in the visible (input) layer
        n_hidden: int
            number of units in the hidden layer
        learning_rate: float
            controlling speed of the contrastive divergence update
        initial_weights: dict
            with keys 'weights', 'bias_v', 'bias_h' and values being array-
             like objects of shapes (n_visible, n_hidden), (n_visible,), and
             (n_hidden,) respectively, containing initial values of
             bidirectional weights, visible and hidden layer biases. If None
             the weights are initialized randomly. Default is None.

        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.initial_weights = initial_weights

        # Initialize weights
        self.weights, self.bias_v, self.bias_h = self._initialize_weights()

        # Placeholder for the minibatch
        self.input_data = tf.placeholder(tf.float32, [None, self.n_visible])

        # Initialize all operations within the computational graph, more
        # importantly set them as instance attributes. Otherwise calling a
        # method containing tf operation in a loop will result in creating more
        # operations

        # Forward pass ========================================================
        # Feed data forward, compute hidden layer activations
        self.hidden_probs = tf.nn.sigmoid(
            tf.matmul(self.input_data, self.weights) + self.bias_h
            )

        # Sample binary states of hidden units from p(hidden | self.input_data)
        self.hidden_sample = tf.nn.relu(
            tf.sign(self.hidden_probs -
                    tf.random_uniform(tf.shape(self.hidden_probs))
                    )
            )

        # Backward pass =======================================================
        # Pass sample of binary hidden activations back to visible layer
        self.reconstructed_probs = tf.nn.sigmoid(
            tf.matmul(self.hidden_sample, tf.transpose(self.weights)) +
            self.bias_v
            )

        # Reconstruct the input data from p(visible | hidden_sample)
        self.reconstructed_input = tf.nn.relu(
            tf.sign(self.reconstructed_probs -
                    tf.random_uniform(tf.shape(self.reconstructed_probs))
                    )
            )

        # Contrastive divergence update =======================================
        # Compute activations of the hidden layer given the reconstructed data
        self.hidden_1 = tf.nn.sigmoid(
            tf.matmul(self.reconstructed_input, self.weights) + self.bias_h
            )

        # Compute the positive and negative phase gradients
        self.positive_grad = tf.matmul(tf.transpose(self.input_data),
                                       self.hidden_sample)
        self.negative_grad = tf.matmul(tf.transpose(self.reconstructed_input),
                                       self.hidden_1)

        # Compute contrastive divergence
        self.cd = (self.positive_grad - self.negative_grad) / tf.to_float(
            tf.shape(self.input_data)[0])

        # Weight update tf ops
        self.weight_update = tf.assign(
            self.weights, self.weights + self.learning_rate * self.cd
            )

        self.bias_v_update = tf.assign(
            self.bias_v, self.bias_v + self.learning_rate * tf.reduce_mean(
                self.input_data - self.reconstructed_input, 0)
            )

        self.bias_h_update = tf.assign(
            self.bias_h, self.bias_h + self.learning_rate * tf.reduce_mean(
                self.hidden_sample - self.hidden_1, 0)
            )

        # Errors ==============================================================
        # Reconstruction errors below compare the input data versus its
        # backward pass reconstruction represented as binary activations
        # or their probabilities respectively.
        self.error_binary = tf.reduce_mean(
            tf.square(self.input_data - self.reconstructed_input)
            )
        self.error_prob = tf.reduce_mean(
            tf.square(self.input_data - self.reconstructed_probs)
            )

        # Deterministic backward pass computes visible layer activations,
        # without sampling from hidden layers
        self.deterministic_backward_pass = tf.nn.sigmoid(
            tf.matmul(self.hidden_probs, tf.transpose(self.weights)) +
            self.bias_v
            )

        # Attributes to store the current weights and errors as numpy arrays
        self.weights_curr = None
        self.bias_v_curr = None
        self.bias_h_curr = None
        self.reconstruction_error_binary = None
        self.reconstruction_error_prob = None

        # Start the session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def contrastive_divergence_update(self, batch):
        """Perform contrastive divergence iteration and update weights and
        biases. See the tf operations in the 'self.__init__' for more detail.

        Parameters
        ----------
        batch: array-like
            of examples to feed into 'self.input_data' tensor. The shape is
            (?, self.n_visible)

        Returns
        -------
        Nophin'; updates values of 'self.weights', 'self.bias_v', 'self.bias_h'

        """
        self.sess.run(
            [self.weight_update, self.bias_v_update, self.bias_h_update],
            feed_dict={self.input_data: batch}
            )

    def compute_errors(self, batch):
        """Computes reconstruction errors and evaluates them as numpy arrays.

        Parameters
        ----------
        batch: array-like
            of examples to feed into 'self.input_data' tensor. The shape is
            (?, self.n_visible)

        Returns
        -------
        Nophin'

        """
        self.reconstruction_error_binary, self.reconstruction_error_prob = \
            self.sess.run([self.error_binary, self.error_prob],
                          feed_dict={self.input_data: batch})

    def weights_to_numpy(self):
        """Take a snapshot of the weights and biases and store them as numpy
        arrays.

        """
        self.weights_curr, self.bias_v_curr, self.bias_h_curr = self.sess.run(
            [self.weights, self.bias_v, self.bias_h]
            )

    def _initialize_weights(self):
        """Transforms arrays on self.initial_weights into 'tf.Variable'
        instances or initialize the weights randomly, in line with
        Hinton (2010)

        Returns
        -------
        weights: tf.Variable
            of shape (self.n_visible, n_hidden) with initial weight matrix
        bias_v: tf.Variable
            of shape (self.n_visible,) with initial biases of the visible layer
        bias_h: tf.Variable
            of shape (self.n_hidden,) with initial biases of the hidden layer

        """

        if self.initial_weights is None:

            weights = tf.Variable(
                tf.random_normal([self.n_visible, self.n_hidden], stddev=0.01),
                dtype=tf.float32
                )
            bias_v = tf.Variable(np.zeros([self.n_visible,]), dtype=tf.float32)
            bias_h = tf.Variable(np.zeros([self.n_hidden,]), dtype=tf.float32)

        else:

            weights = \
                tf.Variable(self.initial_weights["weights"], dtype=tf.float32)
            bias_v = \
                tf.Variable(self.initial_weights["bias_v"], dtype=tf.float32)
            bias_h = \
                tf.Variable(self.initial_weights["bias_h"], dtype=tf.float32)

        return weights, bias_v, bias_h


def extract_mnist_features_via_rbm(n_hidden, learning_rate=0.1, batch_size=32,
                                   num_epochs=5, input_to_binary=False):
    """Extracts features from the MNIST dataset using a Restricted Boltzmann
    Machine.

    Parameters
    ----------
    n_hidden: int
        number of neurons in the hidden layer
    learning_rate: float
        learning rate for contrastive divergence update
    batch_size: int
        size of the minibatch - number of examples fed into the network per
        contrastive divergence
    num_epochs: int
        numbers of epochs, that is the number of times the entire datasets is
        seen by the model
    input_to_binary: bool
        apply sign() function to the MNIST input, making it binary if True.
        If False input features are real numbers between 0 and 1. Default is
        False

    Returns
    -------
    Nophin' trains an RBM on the MNIST data, printing some outputs and figures

    """
    # Load data into the current directory
    mnist_data = input_data.read_data_sets(".", one_hot=True)

    train_X = mnist_data.train.images

    if input_to_binary:
        train_X = np.sign(train_X)  # every value is in {0, 1}

    # Number of visible units is the length of a flattened image
    n_visible = train_X.shape[1]

    # Instantiate the RBM
    rbm = BinaryRBM(n_visible=n_visible, n_hidden=n_hidden,
                    learning_rate=learning_rate)

    # Train it
    errors_binary = list()
    errors_prob = list()
    for epoch in np.arange(1, num_epochs+1):

        # Cheesily sift through dataset without random sampling of minibatches
        # we need to see results rather soon after all
        for start, end in zip(range(0, train_X.shape[0], batch_size),
                              range(batch_size, train_X.shape[0], batch_size)):

            # Select a batch
            batch_x = train_X[start:end]

            # Perform optimization step
            rbm.contrastive_divergence_update(batch_x)

            if start % 5000 == 0 and start != 0:
                print("Epoch: {}".format(epoch),
                      "Progress: {}%".format(int(start/train_X.shape[0]*1e2)))

                # Compute and store reconstruction errors
                rbm.compute_errors(train_X)
                errors_binary.append(rbm.reconstruction_error_binary)
                errors_prob.append(rbm.reconstruction_error_prob)

        print("\nEpoch {} finished.".format(epoch))
        print("reconstruction error, binary: {}".format(errors_binary[-1]))
        print("reconstruction error, prob: {} \n".format(errors_prob[-1]))

    # Get the matrix of weighs
    rbm.weights_to_numpy()
    weights = rbm.weights_curr

    # Plotting department =====================================================
    # Error vs. training iterations plot
    plt.plot(errors_binary)
    plt.xlabel("Update Step")
    plt.ylabel("Error")
    plt.title("Training Error")

    # Plot features implied by weights of each hidden unit: total of n_hidden
    # images laid out as tile
    pic = Image.fromarray(
        tile_raster_images(X=weights.T, img_shape=(28, 28),
                           tile_shape=(20, 25), tile_spacing=(1, 1))
        )

    fig, pic_ax = plt.subplots(1, 1, figsize=(8, 8))
    pic_plot = pic_ax.imshow(pic)
    pic_plot.set_cmap('gray')
    pic_ax.set_yticklabels([])
    pic_ax.set_xticklabels([])
    pic_ax.set_title("Input features learned by every hidden units")
    fig.tight_layout()

    # Pick a single hidden unit and visualize its connections to the visible
    # layer: get a vector of weights, and transform it to a plottable arrray
    hidden_unit_image = 255 * scale_to_unit_interval(
        weights.T[0].reshape(28, 28)
        )

    pic2 = Image.fromarray(hidden_unit_image)

    fig2, pic_ax = plt.subplots(1, 1, figsize=(8, 8))
    pic_plot = pic_ax.imshow(pic2)
    pic_plot.set_cmap('gray')
    pic_ax.set_yticklabels([])
    pic_ax.set_xticklabels([])
    pic_ax.set_title("Input features learned by the first hidden unit")
    fig.tight_layout()

    # Pick a sample of 5 images from the test set
    test_data = mnist_data.test.images
    idx = np.random.choice(len(test_data), 5)
    test_X = mnist_data.test.images[idx]

    # Corrupt each image by randomly setting 50% of values to zero
    corrupted_X = test_X.copy()
    for img in range(len(corrupted_X)):
        corruption_idx = np.random.choice(corrupted_X.shape[1],
                                          int(corrupted_X.shape[1]/2))
        corrupted_X[img, corruption_idx] = 0

    # Feed the corrupted image to the network and run reconstruction step
    reconstructed_X = rbm.sess.run(rbm.deterministic_backward_pass,
                                   feed_dict={rbm.input_data: corrupted_X})

    # Construct plottable images from vectors
    pic_X = [
        Image.fromarray(255 * scale_to_unit_interval(img.reshape(28, 28))) for
        img in test_X
        ]
    pic_corrupted = [
        Image.fromarray(255 * scale_to_unit_interval(
            img.reshape(28, 28))) for img in corrupted_X
        ]
    pic_reconstructed = [
        Image.fromarray(
            255 * scale_to_unit_interval(img.reshape(28, 28)))
        for img in reconstructed_X
        ]

    # Plot'em
    fig3, axes = plt.subplots(5, 3, sharey=True, figsize=(7, 9))

    for col_num, pics in enumerate([pic_X, pic_corrupted, pic_reconstructed]):
        for row_num, pic in enumerate(pics):
            ax = axes[row_num, col_num]
            img = ax.imshow(pic)
            img.set_cmap("gray")
            ax.set_yticklabels([])
            ax.set_xticklabels([])

    fig3.suptitle("Test Set: \n "
                  "Original -> Corrupted Original -> Reconstructed Image")
    plt.show()


if __name__ == "__main__":
    extract_mnist_features_via_rbm(n_hidden=500, learning_rate=1,
                                   batch_size=100, num_epochs=5,
                                   input_to_binary=False)

