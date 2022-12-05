# This part define three functions
#
#     First for all the Convolutional and Max padding layers
#     Second for blocks that predict each characters
#     Third define and compile the main model

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import *
import keras


def conv_block(inputs: keras.engine.keras_tensor.KerasTensor = None, n_filters: int = 32, actvtn: str = 'relu',
               krnl_size: int = 5, max_pooling: bool = True) -> keras.engine.keras_tensor.KerasTensor:
    """
    Convolutional block

    Arguments:
        inputs -- Input numpy ndarray
        n_filters -- Number of filters for the convolutional layer
        actvtn -- activation function of convolutional layer
        krnl_size -- Kernel size of convolutional layer
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns:
        next_layer --  Next layer
    """
    conv = Conv2D(n_filters,  # Number of filters
                  krnl_size,  # Kernel size
                  activation=actvtn,
                  strides=1,
                  padding='same')(inputs)

    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:

        next_layer = MaxPool2D(pool_size=(2, 2), padding='same')(conv)


    else:
        next_layer = conv

    skip_connection = conv

    return next_layer


def char_block(inputs: keras.engine.keras_tensor.KerasTensor = None, n_units: int = 64, dropout_prob: float = 0.5,
               block_name: str = "character", num_symbols: int = 19) -> keras.engine.keras_tensor.KerasTensor:
    """
    Character block that predicts the digit of CAPTCHA

    Arguments:
        inputs -- Input tensor
        n_units -- Number of units for the first Dense layer
        block_name -- The name of each character
        num_symbols -- Number of symbols
    Returns:
        output --  Next layer and output of this block
    """
    dense = Dense(n_units, activation='relu')(inputs)
    dropout = Dropout(dropout_prob)(dense)
    batchnorm = BatchNormalization()(dropout)
    output = Dense(num_symbols, activation='sigmoid', name=block_name)(batchnorm)

    return output


def define_model(cap_img_shape: tuple) -> keras.engine.functional.Functional:
    """
      Define and compile a model for solving CAPTCHA

      Arguments:
          cap_img_shape -- Shape of CAPTCHA image
      Returns:
          model --  The model or solving CAPTCHA
    """

    input = Input(shape=cap_img_shape)

    conv0 = conv_block(inputs=input, n_filters=8, actvtn='relu', krnl_size=5, max_pooling=True)

    conv1 = conv_block(inputs=conv0, n_filters=16, actvtn='relu', krnl_size=3, max_pooling=True)

    conv2 = conv_block(inputs=conv1, n_filters=32, actvtn='relu', krnl_size=3, max_pooling=True)

    conv3 = conv_block(inputs=conv2, n_filters=64, actvtn='leaky_relu', krnl_size=3, max_pooling=True)

    conv4 = conv_block(inputs=conv3, n_filters=128, actvtn='leaky_relu', krnl_size=3, max_pooling=True)

    flat = Flatten()(conv4)
    dropout = Dropout(0.5)(flat)
    batchnorm = BatchNormalization()(dropout)

    output1 = char_block(inputs=batchnorm, n_units=64, dropout_prob=0.5, block_name="character1")

    output2 = char_block(inputs=batchnorm, n_units=64, dropout_prob=0.5, block_name="character2")

    output3 = char_block(inputs=batchnorm, n_units=64, dropout_prob=0.4, block_name="character3")

    output4 = char_block(inputs=batchnorm, n_units=64, dropout_prob=0.4, block_name="character4")

    output5 = char_block(inputs=batchnorm, n_units=64, dropout_prob=0.5, block_name="character5")

    output = [output1, output2, output3, output4, output5]

    model = Model(inputs=input, outputs=output, name="captcha-recognition-cnn-model")

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model