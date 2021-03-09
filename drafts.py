"""
vae = VariationalAutoencoder(
    input_dim = (28,28,1)
    , encoder_conv_filters = [32,64,64, 64]
    , encoder_conv_kernel_size = [3,3,3,3]
    , encoder_conv_strides = [1,2,2,1]
    , decoder_conv_t_filters = [64,64,32,1]
    , decoder_conv_t_kernel_size = [3,3,3,3]
    , decoder_conv_t_strides = [1,2,2,1]
    , z_dim = 2
    , r_loss_factor = 1000
)

"""

D = {'input_dim' : (28,28,1)
    , 'encoder_conv_filters': [32, 64, 64, 64]
    , 'encoder_conv_kernel_size': [3, 3, 3, 3]
    , 'encoder_conv_strides': [1,2,2,1]
    , 'decoder_conv_t_filters': [64,64,32,1]
    , 'decoder_conv_t_kernel_size': [3,3,3,3]
    , 'decoder_conv_t_strides': [1,2,2,1]
    , 'z_dim': 2
    , 'r_loss_factor': 1000
}

from utils.custom_utils import timer, benchmark
import time

@timer
def fib(n):
    time.sleep(2)
    fib = [0, 1] + [0]*(n-1)
    for i in range(2, n+1):
        fib[i] = fib[i-1] + fib[i-2]
    print(fib[n])
    return fib[n]




fib = benchmark(fib)
fib(100)

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(512, 512, 1)),
    tf.keras.layers.Conv2D(3, 3),
    tf.keras.layers.MaxPool2D(),
])

tf.keras.utils.plot_model(model)