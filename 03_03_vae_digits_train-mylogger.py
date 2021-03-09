# coding: utf-8

# # VAE Training

# ## imports

import os, sys
import tensorflow as tf
import logging


from models.VAE_a import VariationalAutoencoder
from utils.loaders_a import load_mnist
from utils.gpu_utils import gpu_optim
from utils.custom_utils import mk_run_folders, mk_folder
from utils.custom_utils import timer, benchmark, copy_weights, try_func
from utils.my_logger import create_log


###
custom_logger = True
original_stdout = sys.__stdout__ # Save a reference to the original standard output
if custom_logger == True:
    log_path = os.path.join(os.path.dirname(__file__), 'logs')  # logs dir
    logger = create_log('03_03_ref.log', log_path, 20, 40)  # log name, log level, consol level
else:
    pass
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)





logger.info('\n***RUN***\n')

# run params
gpu_flag = True
build_flag = False  # build / load
train_flag = True

SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

mk_run_folders(RUN_FOLDER, 'viz', 'images', 'weights')

if gpu_flag:
    gpu_optim()  # gpu start

# values
EPOCHS = 1
BATCH_SIZE = 16
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0
LEARNING_RATE = 0.0005

# ## data
(x_train, y_train), (x_test, y_test) = load_mnist()


# ## architecture
vae = VariationalAutoencoder(
    input_dim=(28, 28, 1)
    , encoder_conv_filters=[32, 64, 64, 64]
    , encoder_conv_kernel_size=[3, 3, 3, 3]
    , encoder_conv_strides=[1, 2, 2, 1]
    , decoder_conv_t_filters=[64, 64, 32, 1]
    , decoder_conv_t_kernel_size=[3, 3, 3, 3]
    , decoder_conv_t_strides=[1, 2, 2, 1]
    , z_dim=2
    , r_loss_factor=1000
)

vae.compile(LEARNING_RATE)

if build_flag:
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights_'))
    print("wights loaded: ", os.path.join(RUN_FOLDER, 'weights/weights_'))

vae.encoder.summary()
vae.decoder.summary()

logger.debug("\nTraining start? any - yes / 0 - NO")
ask = input()
if ask == 0 or ask =='0':
    sys.exit("STOP execution")


# ## training
logger.info('\n***Training start***\n')

vae.train = timer(vae.train)  # timer decorator

if train_flag:
    vae.train(
        x_train
        , batch_size=BATCH_SIZE
        , epochs=EPOCHS
        , run_folder=RUN_FOLDER
        , print_every_n_batches=PRINT_EVERY_N_BATCHES
        , initial_epoch=INITIAL_EPOCH
    )

logger.info('\n***Training finished***\n')
# copy weights
copy_weights(RUN_FOLDER, add='_')

# # 'application' code - > err message
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warn message')
# logger.error('error message')
# logger.critical('critical message')



tf.keras.backend.clear_session()
sys.stdout = original_stdout # Reset the standard output to its original value