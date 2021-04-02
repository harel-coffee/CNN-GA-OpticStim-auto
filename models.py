import numpy as np
import keras
import tensorflow as tf

from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.models import model_from_json
from keras.optimizers import RMSprop

from keras import activations
from keras import backend as K
from keras import layers
from keras import models

def load_fmnist():
    K.set_image_data_format('channels_last')

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    num_classes = len(np.unique(y_test))
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    print(np.max(x_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('\nLoaded FMNIST dataset')
    print('-----------------------')
    print('dataset specifications')
    print('  * number of classes:\t', num_classes)
    print('  * sample dimensions:\t', (img_rows, img_cols))
    print()
    print('dataset composition')
    print('  * train samples:\t', x_train.shape[0])
    print('  * test samples:\t', x_test.shape[0])
    print()

    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    K.set_image_data_format('channels_last')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_classes = len(np.unique(y_test))
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('\nLoaded MNIST dataset')
    print('-----------------------')
    print('dataset specifications')
    print('  * number of classes:\t', num_classes)
    print('  * sample dimensions:\t', (img_rows, img_cols))
    print()
    print('dataset composition')
    print('  * train samples:\t', x_train.shape[0])
    print('  * test samples:\t', x_test.shape[0])
    print()

    return (x_train, y_train), (x_test, y_test)

def create_LindseyNet(input_shape=(28, 28, 1), N_BN=1, D_VVS=3, scale=1):
    net_input = keras.Input(shape=input_shape, name="img")

    x = layers.Conv2D(
        filters=10*scale,
        kernel_size=9,
        strides=(1, 1),
        input_shape=(32, 32, 1),
        padding='same',
        data_format='channels_last'
    )(net_input)
    x = layers.Activation(activations.relu)(x)
    x = layers.BatchNormalization()(x)

    # RetinaNet layer 2
    # output is given by retinal ganglion cells = optic nerve
    x = layers.Conv2D(
        filters=N_BN*scale,
        kernel_size=9,
        strides=(1, 1),
        padding='same'
    )(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.BatchNormalization()(x)

    # VVSNet
    for i in range(D_VVS):
        x = layers.Conv2D(
            filters=10*scale,
            kernel_size=9,
            strides=(1, 1),
            padding='same'
        )(x)
        x = layers.Activation(activations.relu)(x)
        x = layers.BatchNormalization()(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Dense(10)(x)
    net_output = layers.Activation(activations.softmax)(x)

    net_name = 'lindseyNet_' + str(N_BN) + '_' + str(D_VVS) + '_' + str(scale)
    net = keras.Model(net_input, net_output, name=net_name)
    net.summary()

    rmsprop = RMSprop()
    net.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop,
        metrics=['accuracy']
    )
    return net


def save_model(model, model_name):
    """Save Keras Model to file."""
    model_json = model.to_json()
    with open(model_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(model_name + '.h5')
    print("Saved model to disk")


def load_model(model_name):
    """Load Keras Model from file."""
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + '.h5')
    print('Loaded model from disk')
    return model

def create_input_to_layer_model(model, layer_list=[6, 12]):
    """Create model that predicts optic nerve and cortical activation given
    input image."""
    all_layer_outputs = [layer.output for layer in model.layers]
    layer_outputs = [all_layer_outputs[idx] for idx in layer_list]

    activation_model = models.Model(
        inputs=model.input,
        outputs=layer_outputs
    )  # Creates a model that will return these outputs, given the model input
    return activation_model

def create_layer_to_output_model(model, layer):
    input_shape = model.layers[layer + 1].get_input_shape_at(0)
    input_shape = input_shape[1:]
    layer_input = layers.Input(shape=input_shape)

    # stack all the layers from layer_act to layer_targ
    x = layer_input
    for layer in model.layers[(layer + 1):]:
        x = layer(x)

    # create the model
    new_model = models.Model(layer_input, x)
    return new_model

def create_layer_to_layer_model(model, layer1=6, layer2=12):
    """Create model that predicts cortical activation from optic nerve
    activation."""
    input_shape = model.layers[layer1 + 1].get_input_shape_at(0)  # get the input shape of desired layer
    input_shape = input_shape[1:]  # batch size is automatically added below and should be removed here
    layer_input = layers.Input(shape=input_shape)  # a new input tensor to be able to feed the desired layer

    # stack all the layers from layer_act to layer_targ
    x = layer_input
    for layer in model.layers[(layer1 + 1):(layer2 + 1)]:
        x = layer(x)

    # create the model
    new_model = models.Model(layer_input, x)
    return new_model
