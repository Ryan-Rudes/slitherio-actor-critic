from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf

def make_actor():
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    observation_input = Input(shape = (84, 84, 3))
    length_input = Input(shape = (1,))
    x = Conv2D(32, 8, 4, activation = 'relu')(observation_input)
    x = Conv2D(64, 4, 2, activation = 'relu')(x)
    x = Conv2D(64, 3, 1, activation = 'relu')(x)
    x = Flatten()(x)
    x = Concatenate()([x, length_input])
    x = Dense(512, activation = 'relu')(x)
    action = Dense(2, activation = 'sigmoid', kernel_initializer = last_init)(x)
    actor = Model([observation_input, length_input], action)
    return actor

def make_critic():
    observation_input = Input(shape = (84, 84, 3))
    length_input = Input(shape = (1,))
    action_input = Input(shape = (2,))

    x = Conv2D(32, 8, 4, activation = 'relu')(observation_input)
    x = Conv2D(64, 4, 2, activation = 'relu')(x)
    x = Conv2D(64, 3, 1, activation = 'relu')(x)
    x = Flatten()(x)
    x = Concatenate()([x, length_input])
    y = Concatenate()([x, action_input])

    x = Dense(512, activation = 'relu')(x)
    y = Dense(512, activation = 'relu')(y)

    V = Dense(128, activation = 'relu')(x)
    V = Dense(128, activation = 'relu')(V)
    V = Dense(1)(V)

    Q = Dense(128, activation = 'relu')(y)
    Q = Dense(128, activation = 'relu')(Q)
    Q = Dense(1)(Q)

    A = Q - V
    Q = Q + A

    critic = Model([observation_input, length_input, action_input], Q)
    return critic