import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.preprocessing import image
import random
import time

from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Activation, ZeroPadding2D
from keras.layers import add, Flatten
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,BatchNormalization,GlobalMaxPooling2D
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K 
from keras.regularizers import l2
from keras_applications import resnet
import os
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Global Constants
NB_CLASS=17
IM_WIDTH=224
IM_HEIGHT=224
train_root='/home/vivo3/CAM/delicate_erdram/train/'
vaildation_root='/home/vivo3/CAM/delicate_erdram/test/'
test_root='/home/vivo3/CAM/delicate_erdram/test/'
batch_size=16
EPOCH=40

# train data
train_datagen = ImageDataGenerator(
    #rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)
train_generator = train_datagen.flow_from_directory(
    train_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    #save_to_dir='/home/vivo3/CAM/delicate_erdram/train_gene/',
    shuffle=True
)

# vaild data
vaild_datagen = ImageDataGenerator(
    #rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)
vaild_generator = train_datagen.flow_from_directory(
    vaildation_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    #save_to_dir='/home/vivo3/CAM/delicate_erdram/test_gene/',
)

# test data
test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = train_datagen.flow_from_directory(
    test_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
)

def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{epoch:03d}-{acc:.3f}.hdf5")

def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def check_print():
    base_model = resnet.ResNet101(include_top=False,weights='imagenet',input_shape=(224, 224, 3),classes=17,backend=keras.backend,
                                   layers=keras.layers,models=keras.models,utils=keras.utils)
    base_model.trainable=False
    x = base_model.output
    #x = GlobalMaxPooling2D()(x)
    #x = Dense(2048, activation='relu')(x)
    #x = BatchNormalization()(x)
    x = Flatten()(x)
    #x = Dropout(0.5)(x)
    #x = Dense(2048,activation='relu')(x)
    x = Dropout(0.5)(x)
    #x = Dense(2048,activation='relu',kernel_regularizer=l2(0.0003))(x)
    predictions = Dense(NB_CLASS, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    model.compile(optimizer=Adam(lr=0.00005, beta_1=0.9,beta_2=0.99,epsilon=1e-08,decay=1e-6),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model
    
if __name__ == '__main__':
    
    model=check_print()
    save_dir = make_save_dir('/home/vivo3/CAM/model/', 'erdram')
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)
    early_stopping = EarlyStopping(monitor='acc', patience=5, verbose=2)
    model.fit_generator(train_generator,validation_data=vaild_generator,epochs=EPOCH,steps_per_epoch=train_generator.n/batch_size,validation_steps=vaild_generator.n/batch_size,callbacks=[checkpointer,early_stopping])
    model.save('/home/vivo3/CAM/model/20191031resnet_101_delicate.h5')
    loss,acc,top_acc=model.evaluate_generator(test_generator, steps=test_generator.n/batch_size)
    print ('Test result:loss:%f,acc:%f,top_acc:%f' % (loss, acc, top_acc))
