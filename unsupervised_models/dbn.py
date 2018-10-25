from unsupervised_models.rbm import BinaryRBM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
from theano_utils import tile_raster_images
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

"""Constructs and trains a deep belief network for the MNIST dataset. First,
pre-trains a stack of restricted Boltzmann machines, then fine-tunes the 
parameters in an MLP. 

The exercise is similar to that in Hinton and Salakhutdinov, Science (2006).

"""


def rbm_greedy_pre_training(training_data, rbm_hidden_layers, rbm_kwargs,
                            train_rbm_kwargs):
    """Performs greedy layer-wise pre-training of a stack of restricted
    Boltzmann machines. The first RBM is trained on the 'training_data'. After
    that its weights are frozen and the second RBM is trained on the sigmoid
    activations of the first RBM. This procedure repeats for all subsequent
    layers.

    Parameters
    ----------
    training_data: np.array
        of shape (# examples, rbm.n_hidden) of examples to train the first
        restricted Boltzmann machine
    rbm_hidden_layers: list
        of ints, where each element corresponds to the number of hidden units
        in layers of RBMs
    rbm_kwargs: list
        of dicts with 'learning_rate' and 'initial_weights' arguments for
        instantiation of 'BinaryRBM'; the 'n_visible' and 'n_hidden' arguments,
         are determined endogenously to construct the stacked net. Each element
         in the list corresponds to the element in 'hidden_layers'
    train_rbm_kwargs: list
        of dictionaries of keyword arguments of the 'train_rbm' function for
        networks corresponding to the elements in 'rbm_hidden_layers'

    Returns
    -------

    """

    # Greedily pre-train models layer-wise, storing BinaryRBM instances
    rbm_models = []
    errors_binary = []  # reconstruction error input vs. binary activations
    errors_prob = []    # reconstruction error input vs. sigmoid pobs

    for model_number, n_hidden in enumerate(rbm_hidden_layers):

        print("Training RBM layer {} out of {}".format(
            model_number+1, len(rbm_hidden_layers)))

        # Unpack parameters
        these_model_kwargs = rbm_kwargs[model_number]
        these_training_kwargs = train_rbm_kwargs[model_number]

        if model_number == 0:
            # The first model trains on the input data
            n_visible = training_data.shape[1]

            # Initialize and train the model
            this_model = BinaryRBM(n_visible=n_visible, n_hidden=n_hidden,
                                   **these_model_kwargs)

            this_model, these_errors_binary, these_errors_prob = \
                train_rbm(this_model, training_data, **these_training_kwargs)

            # Feed forward the entire training set to the hidden layer,
            # generating input for the next RBM layer
            new_input = this_model.sess.run(
                this_model.hidden_probs,
                feed_dict={this_model.input_data: training_data}
                )

        else:
            # Train the subsequent RBMs on the previous models' outputs
            this_model = BinaryRBM(
                n_visible=rbm_models[model_number-1].n_hidden,
                n_hidden=n_hidden, **these_model_kwargs)

            this_model, these_errors_binary, these_errors_prob = \
                train_rbm(this_model, new_input, **these_training_kwargs)

            # Feed forward the entire training set to the hidden layer,
            # generating input for the next RBM layer
            new_input = this_model.sess.run(
                this_model.hidden_probs,
                feed_dict={this_model.input_data: new_input}
                )

        # Append the output
        rbm_models.append(this_model)
        errors_binary.append(these_errors_binary)
        errors_prob.append(these_errors_prob)

    return rbm_models, errors_binary, errors_prob


def train_rbm(rbm, training_data, batch_size=32, num_epochs=5,
              compute_errors_every_x_updates=None,
              print_progress_every_x_updates=None):
    """Trains a Restricted Boltzmann Machine, storing the reconstruction errors
    during training process if needed. Also prints training progress upon
    request.

    Parameters
    ----------
    rbm: BinaryRBM
        instance; a restricted Boltzmann machine
    training_data: np.array
        of shape (# examples, rbm.n_hidden) of examples to train 'rbm'
    batch_size: int
        size of the minibatch - number of examples fed into the network per
        contrastive divergence. Default is 32
    num_epochs: int
        numbers of epochs, that is the number of times the entire datasets is
        seen by the model. Default is 5
    compute_errors_every_x_updates: int or None
        number of contrastive divergence updates before between appending
        training errors to the output. If None, the errors are not computed.
        Default is None
    print_progress_every_x_updates: int or None
        print training progress message every 'verbose' number of batches. If
        None don't print anything. Default is None

    Returns
    -------
    rmb: BinaryRBM
        instance after training
    errors_binary: np.array
        of errors comparing training_data to binary activations of the visible
        layer units during backward pass. See 'BinaryRBM.reconstructed_input'
        attribute for details
    errors_prob: np.array
        of errors comparing training_data to sigmoid probabilities of
        activations of the visible layer units during backward pass. See
        'BinaryRBM.reconstructed_probs' attribute for details

    """
    errors_binary = list()
    errors_prob = list()
    compute_error_counter = 0
    verbosity_counter = 0

    for epoch in np.arange(1, num_epochs+1):

        for start, end in \
            zip(range(0, training_data.shape[0], batch_size),
                range(batch_size, training_data.shape[0], batch_size)):

            # Select a batch
            batch = training_data[start:end]

            # Perform optimization step
            rbm.contrastive_divergence_update(batch)

            # Adjust error counter
            compute_error_counter += 1

            if compute_error_counter == compute_errors_every_x_updates:
                # Compute errors
                rbm.compute_errors(training_data)
                errors_binary.append(rbm.reconstruction_error_binary)
                errors_prob.append(rbm.reconstruction_error_prob)

                # Reset the counter
                compute_error_counter = 0

            # Adjust verbosity counter
            verbosity_counter += 1

            # Print progress
            if verbosity_counter == print_progress_every_x_updates:

                print("Epoch: {}".format(epoch),
                      "Progress: {}%".format(
                          int(start/training_data.shape[0]*1e2))
                      )

                verbosity_counter = 0  # reset the counter

        print("\nEpoch {} finished.".format(epoch))

        if compute_errors_every_x_updates is not None:
            print("reconstruction error, binary: {}".format(errors_binary[-1]))
            print("reconstruction error, prob: {} \n".format(errors_prob[-1]))

    return rbm, np.array(errors_binary), np.array(errors_prob)


def construct_fine_tuning_mlp(rbms, last_layers="mnist", **compile_kwargs):
    """Unfolds a list of restricted Boltzmann machines into multilayer
    perceptron and compiles a keras sequential model.

    Parameters
    ----------
    rbms: list
        of BinaryRBM instances containing trained RBMs to unfold into MLP
    last_layers: list of keras.layer instances or string
        the last layers of the MLP to be used for supervised fine-tuning. If
        'mnist' creates a softmax layer with ten units for the generic MNIST
        task. Default is 'mnist'
    compile_kwargs: dict
        of keyword arguments to the keras.models.Sequential.compile() method:
        loss, optimizer, metrics etc.

    Returns
    -------
    mlp: keras.models.Sequential
        instance with stack of input RBMs unfolded into MLP

    """
    # Create a model
    mlp = Sequential()

    # Loop over RBMs unfolding them into perceptron layers
    for model_number, rbm in enumerate(rbms):
        mlp.add(Dense(input_shape=[rbm.n_visible], units=rbm.n_hidden,
                      weights=[rbm.weights_curr, rbm.bias_h_curr],
                      activation="sigmoid",
                      name="rbm_layer_"+str(model_number+1))
                )

    # Add the last layer on top
    if last_layers is "mnist":
        mlp.add(Dense(input_shape=[rbms[-1].n_hidden], units=10,
                      activation="softmax", name="output_softmax_layer")
                )
    else:
        for layer in last_layers:
            mlp.add(layer)

    # Compile and return
    mlp.compile(**compile_kwargs)

    return mlp


def main():
    # Load the data
    mnist_data = input_data.read_data_sets(".", one_hot=True)
    train_X = mnist_data.train.images
    train_y = mnist_data.train.labels
    val_X = mnist_data.validation.images
    val_y = mnist_data.validation.labels
    test_X = mnist_data.test.images
    test_y = mnist_data.test.labels

    # Settings department =====================================================
    # Network architecture: stack of two RBMs
    dbn_rbm_layers = [500, 1000]
    num_rbm_layers = len(dbn_rbm_layers)

    # Settings for each machine:
    rbm_1_settings = {"learning_rate": 0.1}
    rbm_2_settings = {"learning_rate": 0.01}  # hold your horses
    rbm_settings = [rbm_1_settings, rbm_2_settings]

    # Train both machines with the same minibatch size and number of epochs
    rbm_training_settings = [
        {"batch_size": 32, "num_epochs": 5,
         "compute_errors_every_x_updates": 100,
         "print_progress_every_x_updates": 100}
        ] * num_rbm_layers

    # Settings for the fine-tuning stage
    mlp_last_layer = "mnist"  # construct the default output layer for mnist

    # keras.models.Sequential.compile() settings
    mlp_compile_kwargs = {"loss": "binary_crossentropy",
                          "optimizer": Adam(lr=1e-4),  # fine-tune slowly
                          "metrics": ["accuracy"]}

    # keras.callbacks for fitting mlp, monitoring validation accuracy
    early_stopping = EarlyStopping(monitor="val_acc", min_delta=3e-4,
                                   patience=3, mode="max", verbose=2)

    # keras.models.Sequential.fit() settings
    mlp_fit_kwargs = {"epochs": 50,
                      "batch_size": 100,
                      "shuffle": False,
                      "callbacks": [early_stopping]}

    # Greedy pre-training and fine tuning =====================================
    # Sequentially train the RBM layers
    rbms, err_bin, err_prob = rbm_greedy_pre_training(
        training_data=train_X, rbm_hidden_layers=dbn_rbm_layers,
        rbm_kwargs=rbm_settings,
        train_rbm_kwargs=rbm_training_settings)

    # Get the weights and biases as numpy arrays
    [rbm.weights_to_numpy() for rbm in rbms]

    # Unfold into an MLP, print summary, and fit
    mlp = construct_fine_tuning_mlp(rbms, last_layers=mlp_last_layer,
                                    **mlp_compile_kwargs)

    print("\nDone with the RBM pre-training. Unfolding into the following MLP "
          "for fine-tuning:\n")
    mlp.summary()

    mlp.fit(x=train_X, y=train_y, validation_data=(val_X, val_y), verbose=2,
            **mlp_fit_kwargs)

    # Test results ============================================================
    y_pred = mlp.predict(test_X)
    y_pred = np.argmax(y_pred, axis=1)  # transform to numeric labels
    y_true = np.argmax(test_y, axis=1)

    test_accuracy_score = accuracy_score(y_true, y_pred)

    print("Classification accuracy on the test set is {}%".format(
        test_accuracy_score*1e2))

    # Plotting ================================================================
    # Get the weight matrices from the rbm and fine-tuned MLP take the dot
    # product between first and second layers to represent learned features
    weights_rbm_1 = rbms[0].weights_curr
    weights_rbm_2 = weights_rbm_1.dot(rbms[1].weights_curr)
    weights_mlp_1 = mlp.layers[0].get_weights()[0]
    weights_mlp_2 = weights_mlp_1.dot(mlp.layers[1].get_weights()[0])

    # Construct the images
    images = []
    for weights in [weights_rbm_1, weights_rbm_2, weights_mlp_1,
                    weights_mlp_2]:
        images.append(
            Image.fromarray(
                tile_raster_images(X=weights.T, img_shape=(28, 28),
                                   tile_shape=(7, 7), tile_spacing=(1, 1)))
            )

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), facecolor="white")

    for ax_num, ax in enumerate(axes.flatten()):
        img = ax.imshow(images[ax_num])
        img.set_cmap("gray")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    axes[0, 0].set_title("First Layer Features", fontsize=14)
    axes[0, 0].set_ylabel("Pre-Trained RBMs", fontsize=14)
    axes[0, 1].set_title("Second Layer Features", fontsize=14)
    axes[1, 0].set_ylabel("After MLP Fine-Tuning", fontsize=14)

    fig.suptitle("Extracted Features from the First 49 Units",
                 fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    main()
