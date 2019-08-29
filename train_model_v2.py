from model_v3 import  *
from data import *
import keras
import cv2
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

train_path = "640x400_all"
image_folder = "train"
label_folder = "train_labels"
valid_path =  "640x400_all"
valid_image_folder ="val"
valid_label_folder = "val_labels"
test_path = '640x400_all/test/'
log_filepath = './log'
flag_multi_class = True
num_classes = 4
batch_size = 15

dp = data_preprocess(
    train_path=train_path,image_folder=image_folder,label_folder=label_folder,
    valid_path=valid_path,valid_image_folder=valid_image_folder,valid_label_folder=valid_label_folder,
    flag_multi_class=flag_multi_class, num_classes=num_classes, data_gen_args=dict(),
    test_path=test_path, img_rows=640, img_cols=400, csv_name='class_dict.csv'
)

# train your own model
train_data = dp.trainGenerator(batch_size)
valid_data = dp.validLoad(batch_size)
test_data = dp.testGenerator()

model = unet(num_class=4, input_size=(640, 400, 3))

tb_cb = TensorBoard(log_dir=log_filepath)
val_loss_model_checkpoint = keras.callbacks.ModelCheckpoint(
    './model/val_loss_adadelta_categorical_crossentropy_100_epochs_1k_steps_8_redux.hdf5',
	monitor='val_loss', verbose=1, save_best_only=True
)
val_acc_model_checkpoint = keras.callbacks.ModelCheckpoint(
    './model/val_acc_adadelta_categorical_crossentropy_100_epochs_1k_steps_8_redux.hdf5', 
	monitor='val_acc', verbose=1, save_best_only=True
)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
# https://keras.io/callbacks/#reducelronplateau

history = model.fit_generator(
    train_data,
    steps_per_epoch=1000, epochs=100,
    validation_steps=2265//batch_size, # 2265 validation images / 15 = 151
	# """validation_steps: Only relevant if validation_data is a generator. 
	# Total number of steps (batches of samples) to yield from validation_data generator
	#  before stopping at the end of every epoch. 
	# It should typically be equal to the number of samples of your 
	# validation dataset divided by the batch size. 
	# Optional for Sequence: if unspecified, 
	# will use the len(validation_data) as a number of steps."""
    validation_data=valid_data,
	shuffle=True,
    callbacks=[val_loss_model_checkpoint, val_acc_model_checkpoint, tb_cb]
)
