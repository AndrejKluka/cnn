import time
start = time.perf_counter()
import functions as fn
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from keras import models
import os



#set up paths
modelpath='my_model.h5'
dirname = os.path.dirname(os.path.abspath(__file__))
data_dir= os.path.join(dirname, 'data')

train_pics_dir=os.path.join(data_dir, 'train_pics')
train_annot_dir=os.path.join(data_dir, 'train_annot')

eval_pics_dir=os.path.join(data_dir, 'eval_pics')
eval_annot_dir=os.path.join(data_dir, 'eval_annot')


# Load data and model
#x_train_data=np.load(os.path.join(data_dir,'x_train_data.npy'))
#y_train_data=np.load(os.path.join(data_dir,'y_train_data.npy'))

x_eval_data=np.load(os.path.join(data_dir,'x_eval_data.npy'))
y_eval_data=np.load(os.path.join(data_dir,'y_eval_data.npy'))

get_custom_objects().update({"custom_loss": fn.custom_loss})
model=models.load_model(modelpath)

stop = time.perf_counter()
print(int((stop - start) * 100) / 100., 'sec -- shit loaded and prepared \n')
y_pred = model.predict(x_eval_data)
stop1 = time.perf_counter()
print(int((stop1 - stop) * 100) / 100., 'sec -- model inference runtime for evaluation dataset (',\
      int((stop1 - stop)/x_eval_data.shape[0] * 100) / 100.,'sec per img)\n')


# Get object confidence values
num_planes= np.sum(y_eval_data[:,:,:,0])

true_conf= np.sum(y_eval_data[:,:,:,0]) / (y_eval_data.shape[0]* y_eval_data.shape[1]* y_eval_data.shape[2])
pred_conf= np.sum(y_pred[:,:,:,0]) / (y_pred.shape[0]* y_pred.shape[1]* y_pred.shape[2])
print(true_conf,'= true average confidence  vs \n',pred_conf,'= predicted average confidence')

#true_cen= np.sum(y_eval_data[:,:,:,0]) / (y_eval_data.shape[0]* y_eval_data.shape[1]* y_eval_data.shape[2])
#pred_conf= np.sum(y_pred[:,:,:,0]) / (y_pred.shape[0]* y_pred.shape[1]* y_pred.shape[2])




''' Plot predictions in pictures,   Red=true   Green=prediction'''
img = os.listdir(eval_pics_dir)
fn.viz_output(model, img[1],  t=3,  threshold=0.3,  plot=True,  pic_dir= eval_pics_dir,  annot_dir= eval_annot_dir)


#x_data, OGimg = prep_pic(img[2])
#y_pred = model.predict(x_data)
#pred_bbs = ypred_to_bbs(y_pred[0], OGimg.shape, threshold=0.05)
#print(pred_bbs)

#viz_output(model, img[1001], threshold=0.9995, plot=True)
#cen=fn.eval_centers(model, threshold=0.99)

















stop = time.perf_counter()
print(int((stop - start) * 100) / 100., 'sec -- finished \n')