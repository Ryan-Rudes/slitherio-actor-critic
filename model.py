from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomUniform
import tensorflow as tf
from math import sqrt

def make_actor():
    fan_in_1 = 84 * 84 * 4
    fan_in_2 = 20 * 20 * 32
    fan_in_3 = 9 * 9 * 32
    fan_in_4 = 1
    fan_in_5 = 1576
    fan_in_6 = 200

    init_1 = RandomUniform(minval = -1 / sqrt(fan_in_1), maxval = 1 / sqrt(fan_in_1))
    init_2 = RandomUniform(minval = -1 / sqrt(fan_in_2), maxval = 1 / sqrt(fan_in_2))
    init_3 = RandomUniform(minval = -1 / sqrt(fan_in_3), maxval = 1 / sqrt(fan_in_3))
    init_4 = RandomUniform(minval = -1 / sqrt(fan_in_4), maxval = 1 / sqrt(fan_in_4))
    init_5 = RandomUniform(minval = -1 / sqrt(fan_in_5), maxval = 1 / sqrt(fan_in_5))
    init_6 = RandomUniform(minval = -1 / sqrt(fan_in_6), maxval = 1 / sqrt(fan_in_6))

    observation_input = Input(shape = (84, 84, 4))
    length_input = Input(shape = (1,))
    x = Conv2D(32, 8, 4, activation = 'relu', kernel_initializer = init_1, bias_initializer = init_1)(observation_input)
    x = Conv2D(32, 4, 2, activation = 'relu', kernel_initializer = init_2, bias_initializer = init_2)(x)
    x = Conv2D(32, 3, 1, activation = 'relu', kernel_initializer = init_3, bias_initializer = init_3)(x)
    x = Flatten()(x)
    y = Dense(8, activation = 'relu', kernel_initializer = init_4, bias_initializer = init_4)(length_input)
    x = Concatenate()([x, y])
    x = Dense(200, activation = 'relu', kernel_initializer = init_5, bias_initializer = init_5)(x)
    x = Dense(200, activation = 'relu', kernel_initializer = init_6, bias_initializer = init_6)(x)
    action = Dense(2, activation = 'tanh', kernel_initializer = RandomUniform(minval = -0.0003, maxval = 0.0003), bias_initializer = RandomUniform(minval = -0.0003, maxval = 0.0003))(x)
    actor = Model([observation_input, length_input], action)
    return actor

def make_critic():
    fan_in_1 = 84 * 84 * 4
    fan_in_2 = 20 * 20 * 32
    fan_in_3 = 9 * 9 * 32
    fan_in_4 = 1571
    fan_in_5 = 200

    init_1 = RandomUniform(minval = -1 / sqrt(fan_in_1), maxval = 1 / sqrt(fan_in_1))
    init_2 = RandomUniform(minval = -1 / sqrt(fan_in_2), maxval = 1 / sqrt(fan_in_2))
    init_3 = RandomUniform(minval = -1 / sqrt(fan_in_3), maxval = 1 / sqrt(fan_in_3))
    init_4 = RandomUniform(minval = -1 / sqrt(fan_in_4), maxval = 1 / sqrt(fan_in_4))
    init_5 = RandomUniform(minval = -1 / sqrt(fan_in_5), maxval = 1 / sqrt(fan_in_5))

    observation_input = Input(shape = (84, 84, 4))
    length_input = Input(shape = (1,))
    action_input = Input(shape = (2,))

    x = Conv2D(32, 8, 4, activation = 'relu', kernel_initializer = init_1, bias_initializer = init_1, kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01))(observation_input)
    x = Conv2D(32, 4, 2, activation = 'relu', kernel_initializer = init_2, bias_initializer = init_2, kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01))(x)
    x = Conv2D(32, 3, 1, activation = 'relu', kernel_initializer = init_3, bias_initializer = init_3, kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01))(x)
    x = Flatten()(x)
    x = Concatenate()([x, length_input, action_input])
    x = Dense(200, activation = 'relu', kernel_initializer = init_4, bias_initializer = init_4, kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01))(x)
    x = Dense(200, activation = 'relu', kernel_initializer = init_5, bias_initializer = init_5, kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01))(x)
    Q = Dense(1, kernel_regularizer = l2(0.01), bias_regularizer = l2(0.01), kernel_initializer = RandomUniform(minval = -0.0003, maxval = 0.0003), bias_initializer = RandomUniform(minval = -0.0003, maxval = 0.0003))(x)

    critic = Model([observation_input, length_input, action_input], Q)
    return critic
