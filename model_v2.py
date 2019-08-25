from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf

def unet(pretrained_weights=None, input_size=(512, 512, 3),num_class=2):
	

	def iou(y_true, y_pred, label):
		"""
		Return the Intersection over Union (IoU) for a given label.
		Args:
			y_true: the expected y values as a one-hot
			y_pred: the predicted y values as a one-hot or softmax output
			label: the label to return the IoU for
		Returns:
			the IoU for the given label
		"""
		# extract the label values using the argmax operator then
		# calculate equality of the predictions and truths to the label
		y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
		y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
		# calculate the |intersection| (AND) of the labels
		intersection = K.sum(y_true * y_pred)
		# calculate the |union| (OR) of the labels
		union = K.sum(y_true) + K.sum(y_pred) - intersection
		# avoid divide by zero - if the union is zero, return 1
		# otherwise, return the intersection over union
		return K.switch(K.equal(union, 0), 1.0, intersection / union)

	def mean_iou(y_true, y_pred):
		"""
		Return the Intersection over Union (IoU) score.
		Args:
			y_true: the expected y values as a one-hot
			y_pred: the predicted y values as a one-hot or softmax output
		Returns:
			the scalar IoU value (mean over all labels)
		"""
		# get number of labels to calculate IoU for
		num_labels = K.int_shape(y_pred)[-1]
		# initialize a variable to store total IoU in
		total_iou = K.variable(0)
		# iterate over labels to calculate IoU for
		for label in range(num_labels):
			total_iou = total_iou + iou(y_true, y_pred, label)
		# divide total IoU by number of labels to get mean IoU
		return total_iou / num_labels
	
	def iou_loss_score(true, pred):  # this can be used as a loss if you make it negative
		intersection = true * pred
		notTrue = 1 - true
		union = true + (notTrue * pred)

		return -(K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())
	
	inputs = Input(input_size)
	redux = 4
	conv1 = Conv2D(64//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1 = Conv2D(64//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(128//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(128//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(256//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(256//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(512//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(512//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	# conv5 = Conv2D(1024//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	# conv5 = Conv2D(1024//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
	# drop5 = Dropout(0.5)(conv5)

	conv5 = Conv2D(512//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	conv5 = Conv2D(512//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512//redux, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(drop5))

	merge6 = concatenate([drop4, up6], axis=3)
	conv6 = Conv2D(512//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	conv6 = Conv2D(512//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = Conv2D(256//redux, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv6))
	merge7 = concatenate([conv3, up7], axis=3)
	conv7 = Conv2D(256//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	conv7 = Conv2D(256//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = Conv2D(128//redux, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv7))
	merge8 = concatenate([conv2, up8], axis=3)
	conv8 = Conv2D(128//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	conv8 = Conv2D(128//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = Conv2D(64//redux, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv8))
	merge9 = concatenate([conv1, up9], axis=3)
	conv9 = Conv2D(64//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	conv9 = Conv2D(64//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv9 = Conv2D(64//redux, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	
	if num_class == 2:
		conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
		loss_function = 'binary_crossentropy'
	else:
		conv10 = Conv2D(3, 1, activation='softmax')(conv9)
		loss_function = 'categorical_crossentropy'
	
	model = Model(input=inputs, output=conv10)
	# model.compile(optimizer=Adam(lr=1e-4), loss=loss_function, metrics=["accuracy"])
	model.compile(optimizer=Adam(lr=1e-4), loss=iou_loss_score, metrics=["accuracy"])
	# model.compile(optimizer=SGD(lr=1e-4), loss=loss_function, metrics=["accuracy"])
	model.summary()

	if (pretrained_weights):
		model.load_weights(pretrained_weights)

	return model
