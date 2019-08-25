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

train_path = "640x400_small"
image_folder = "train"
label_folder = "trainannot"
valid_path =  "640x400_small"
valid_image_folder ="val"
valid_label_folder = "valannot"
test_path = '640x400_small/test/'
log_filepath = './log'
flag_multi_class = True
num_classes = 4

dp = data_preprocess(
    train_path=train_path,image_folder=image_folder,label_folder=label_folder,
    valid_path=valid_path,valid_image_folder=valid_image_folder,valid_label_folder=valid_label_folder,
    flag_multi_class=flag_multi_class, num_classes=num_classes, 
    test_path=test_path, img_rows=640, img_cols=400, 
)

# train your own model
train_data = dp.trainGenerator(batch_size=10)
valid_data = dp.validLoad(batch_size=10)
test_data = dp.testGenerator()

model = unet(num_class=4, input_size=(640, 400, 3))

tb_cb = TensorBoard(log_dir=log_filepath)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    './model/Opends_model_v1.hdf5', monitor='val_loss', verbose=1, save_best_only=True
)

history = model.fit_generator(
    train_data,
    steps_per_epoch=150, epochs=100,
    validation_steps=1,
    validation_data=valid_data,
    callbacks=[model_checkpoint, tb_cb]
)