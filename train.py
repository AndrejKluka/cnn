import numpy as np
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2
from keras import models
from keras import layers
from keras.layers import Input
import os

import functions as fn

dirname = os.path.dirname(os.path.abspath(__file__))
data_dir= os.path.join(dirname, 'data')


model = models.Sequential()
model.add(layers.Conv2D(16, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.1))

filters=[32,64,128,256,512]#,1024]
for fil in filters:
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.Conv2D(512, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(5, 3, strides=(1, 1), padding='same', activation='linear', kernel_regularizer= l2(5e-4)))  
model.add(layers.Activation('sigmoid'))

model.compile('Adam', loss=fn.custom_loss)#  'mean_squared_error'
#model.summary()

x_train_data=np.load(os.path.join(data_dir,'x_train_data.npy'))
y_train_data=np.load(os.path.join(data_dir,'y_train_data.npy'))

model.fit(x=x_train_data, y=y_train_data, batch_size=30, epochs=2, verbose=1, validation_split=0.2, shuffle=True)
#model.save('my_model.h5')