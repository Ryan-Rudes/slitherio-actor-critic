import mahotas
import tensorflow as tf

def preprocess(im):
    return mahotas.resize.resize_rgb_to(im, (84, 84)) / 255.0

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))