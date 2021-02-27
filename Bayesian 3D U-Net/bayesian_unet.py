from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

from groupnorm import GroupNormalization
from utils import normal_prior


def down_stage(inputs, filters, kernel_size=3,
               activation="relu", padding="SAME"):
    conv = Conv2D(filters, kernel_size,
                  activation=activation, padding=padding)(inputs)
    conv = GroupNormalization()(conv)
    conv = Conv2D(filters, kernel_size,
                  activation=activation, padding=padding)(conv)
    conv = GroupNormalization()(conv)
    pool = MaxPooling2D()(conv)
    return conv, pool


def up_stage(inputs, skip, filters, prior_fn, kernel_size=3,
             activation="relu", padding="SAME"):
    up = UpSampling2D()(inputs)
    up = tfp.layers.Convolution2DFlipout(filters, 2,
                                         activation=activation,
                                         padding=padding,
                                         kernel_prior_fn=prior_fn)(up)
    up = GroupNormalization()(up)

    merge = concatenate([skip, up])
    merge = GroupNormalization()(merge)

    conv = tfp.layers.Convolution2DFlipout(filters, kernel_size,
                                           activation=activation,
                                           padding=padding,
                                           kernel_prior_fn=prior_fn)(merge)
    conv = GroupNormalization()(conv)
    conv = tfp.layers.Convolution2DFlipout(filters, kernel_size,
                                           activation=activation,
                                           padding=padding,
                                           kernel_prior_fn=prior_fn)(conv)
    conv = GroupNormalization()(conv)

    return conv


def end_stage(inputs, prior_fn, kernel_size=3,
              activation="relu", padding="SAME"):
    conv = tfp.layers.Convolution2DFlipout(1, kernel_size,
                                           activation=activation,
                                           padding="SAME",
                                           kernel_prior_fn=prior_fn)(inputs)
    conv = tfp.layers.Convolution2DFlipout(1, 1, activation="sigmoid",
                                           kernel_prior_fn=prior_fn)(conv)

    return conv


def bayesian_unet(input_shape=(280, 280, 1), kernel_size=3,
                  activation="relu", padding="SAME", **kwargs):
    prior_std = kwargs.get("prior_std", 1)
    prior_fn = normal_prior(prior_std)

    inputs = Input(input_shape)

    conv1, pool1 = down_stage(inputs, 16,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding=padding)
    conv2, pool2 = down_stage(pool1, 32,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding=padding)
    conv3, pool3 = down_stage(pool2, 64,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding=padding)
    conv4, _ = down_stage(pool3, 128,
                          kernel_size=kernel_size,
                          activation=activation,
                          padding=padding)

    conv5 = up_stage(conv4, conv3, 64, prior_fn,
                     kernel_size=kernel_size,
                     activation=activation,
                     padding=padding)
    conv6 = up_stage(conv5, conv2, 32, prior_fn,
                     kernel_size=kernel_size,
                     activation=activation,
                     padding=padding)
    conv7 = up_stage(conv6, conv1, 16, prior_fn,
                     kernel_size=kernel_size,
                     activation=activation,
                     padding=padding)

    conv8 = end_stage(conv7, prior_fn,
                      kernel_size=kernel_size,
                      activation=activation,
                      padding=padding)

    return Model(inputs=inputs, outputs=conv8)
