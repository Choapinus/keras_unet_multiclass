from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from utils import helpers
import cv2
import warnings

warnings.filterwarnings("ignore")

class DataPreprocess:
	def __init__(
		self, db_path=None, save_path=None, csv_name='class_dict.csv',
		img_rows=512, img_cols=512, num_classes = 2, 
		data_gen_args = dict(
			rotation_range=0.2, width_shift_range=0.05,
			height_shift_range=0.05, shear_range=0.05,
			zoom_range=0.05, vertical_flip=True,
			horizontal_flip=True, fill_mode='nearest'
		)):
		
		self.db = db_path
		self.img_rows = img_rows
		self.img_cols = img_cols
		self.train_path = train_path
		self.image_folder = image_folder
		self.label_folder = label_folder
		self.valid_path = valid_path
		self.valid_image_folder = valid_image_folder
		self.valid_label_folder = valid_label_folder
		self.test_path = test_path
		self.test_label_folder = test_label_folder
		self.save_path = save_path
		self.class_names_list, self.label_values = helpers.get_label_info(os.path.join(self.train_path, csv_name))
		self.COLOR_DICT = np.array(self.label_values)
		self.data_gen_args = data_gen_args
		self.image_color_mode = "rgb"
		self.label_color_mode = "rgb"
		self.flag_multi_class = True if len(self.class_names_list) > 2 else False
		self.num_class = num_classes
		self.target_size = (img_rows, img_cols)
		self.img_type = 'png'

	def adjustData(self, img, label):
		if (self.flag_multi_class):
			img = np.float32(img) / 255.
			label = helpers.one_hot_it(label=label, label_values=self.label_values)
		elif (np.max(img) > 1):
			img = img / 255.
			label = label / 255.
			label[label > 0.5] = 1
			label[label <= 0.5] = 0
		
		return (img, label)

	def trainGenerator(self, batch_size, image_save_prefix="image", label_save_prefix="label",
					   save_to_dir=None, seed=7):
		'''
		Can generate image and label at the same time
		use the same seed for image_datagen and label_datagen to ensure the transformation for image and label is the same
		if you want to visualize the results of generator, set save_to_dir = "your path"
		'''
		image_datagen = ImageDataGenerator(**self.data_gen_args)
		label_datagen = ImageDataGenerator(**self.data_gen_args)
		
		image_generator = image_datagen.flow_from_directory(
			self.db,
			classes=['train'],
			class_mode=None,
			color_mode=self.image_color_mode,
			target_size=self.target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=image_save_prefix,
			seed=seed
		)
		
		label_generator = label_datagen.flow_from_directory(
			self.train_path,
			classes=['train_labels'],
			class_mode=None,
			color_mode=self.label_color_mode,
			target_size=self.target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=label_save_prefix,
			seed=seed
		)
		
		train_generator = zip(image_generator, label_generator)
		
		for (img, label) in train_generator:
			img, label = self.adjustData(img, label)
			yield (img, label)

	def testGenerator(self):
		filenames = os.listdir(self.test_path)
		for filename in filenames:
			img = io.imread(os.path.join(self.test_path, filename), as_gray=False)
			img = img / 255.
			img = trans.resize(img, self.target_size, mode='constant')
			# img = np.reshape(img, img.shape + (1,)) if (not self.flag_multi_class) else img
			img = np.reshape(img, (1,) + img.shape)
			yield img
	
	def labelTestGenerator(self):
		filenames = os.listdir(self.test_label_folder)
		for filename in filenames:
			img = io.imread(os.path.join(self.test_label_folder, filename), as_gray=False)
			img = img / 255.
			img = trans.resize(img, self.target_size, mode='constant')
			# img = np.reshape(img, img.shape + (1,)) if (not self.flag_multi_class) else img
			img = np.reshape(img, (1,) + img.shape)
			yield img

	def validLoad(self, batch_size,seed=7):
		image_datagen = ImageDataGenerator(**self.data_gen_args)
		label_datagen = ImageDataGenerator(**self.data_gen_args)
		image_generator = image_datagen.flow_from_directory(
			self.valid_path,
			classes=[self.valid_image_folder],
			class_mode=None,
			color_mode=self.image_color_mode,
			target_size=self.target_size,
			batch_size=batch_size,
			seed=seed)
		label_generator = label_datagen.flow_from_directory(
			self.valid_path,
			classes=[self.valid_label_folder],
			class_mode=None,
			color_mode=self.label_color_mode,
			target_size=self.target_size,
			batch_size=batch_size,
			seed=seed)
		train_generator = zip(image_generator, label_generator)
		for (img, label) in train_generator:
			img, label = self.adjustData(img, label)
			yield (img, label)
		# return imgs,labels

	def saveResult(self, npyfile, size, name, threshold=127):
		for i, item in enumerate(npyfile):
			img = item
			img_std = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
			if self.flag_multi_class:
				for row in range(len(img)):
					for col in range(len(img[row])):
						num = np.argmax(img[row][col])
						img_std[row][col] = self.COLOR_DICT[num]
			else:
				for k in range(len(img)):
					for j in range(len(img[k])):
						num = img[k][j]
						if num < (threshold/255.0):
							img_std[k][j] = road
						else:
							img_std[k][j] = BackGround
			img_std = cv2.resize(img_std, size, interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(os.path.join(self.save_path, ("%s_predict." + self.img_type) % (name)), img_std)
