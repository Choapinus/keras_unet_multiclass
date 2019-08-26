from model_v2 import  *
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

dp = data_preprocess(
    train_path=train_path,image_folder=image_folder,label_folder=label_folder,
    valid_path=valid_path,valid_image_folder=valid_image_folder,valid_label_folder=valid_label_folder,
    flag_multi_class=flag_multi_class, num_classes=num_classes, 
    test_path=test_path, img_rows=640, img_cols=400, csv_name='class_dict.csv'
)

# train your own model
train_data = dp.trainGenerator(batch_size=20)
valid_data = dp.validLoad(batch_size=20)
test_data = dp.testGenerator()

model = unet(num_class=4, input_size=(640, 400, 3))

tb_cb = TensorBoard(log_dir=log_filepath)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    './model/model_v2_rmsprop_categorical_crossentropy.hdf5', monitor='val_loss', verbose=1, save_best_only=True
)

history = model.fit_generator(
    train_data,
    steps_per_epoch=1000, epochs=100,
    validation_steps=1,
    validation_data=valid_data,
	shuffle=True,
    callbacks=[model_checkpoint, tb_cb]
)
