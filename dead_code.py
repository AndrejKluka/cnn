"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""

#import subprocess, os, paramiko
import functions as fn
import pretraining, do
import adjusted_callbacks as fwc
import cv2, os
import numpy as np
import time
import filter_viz
start = fn.time_print(None, None, None)
import matplotlib.pyplot as plt
#from keras.utils.generic_utils import get_custom_objects
#from keras import models
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate, interpolation
import keras
from keras import layers
from keras import models
from keras.applications import vgg16
stop = fn.time_print(start, None, 'Modules imported')
import scipy.misc
from keras.regularizers import l2
import json

****************************************************************** SCRIPT CODE

def run_file_script(file):
	ssh_client = fn.setup_ssh()
	stdin, stdout, stderr = ssh_client.exec_command('cd ~/env/keras/bin; echo $PWD')	#activates necesarry python env
	for linee in stdout.readlines(): print(linee)
	for line in stderr.readlines(): print(line)
	stdin, stdout, stderr = ssh_client.exec_command('echo $PWD')	#activates necesarry python env
	for linee in stdout.readlines(): print(linee)
	for line in stderr.readlines(): print(line)
	#stdin, stdout, stderr = ssh_client.exec_command('source ~/env/keras/bin/activate_this.py')	#activates necesarry python env
	for line in stdout.readlines(): print(line)
	for line in stderr.readlines(): print(line)
	print('*****************'+'. ~/env/keras/bin/activate ;python {}'.format(file))
	#stdin, stdout, stderr = ssh_client.exec_command('. ~/env/keras/bin/activate; pip list')
	#stdin, stdout, stderr = ssh_client.exec_command('source .bashrc ; ~/env/keras/bin/python {}'.format(file))	#runs python script
	stdin, stdout, stderr = ssh_client.exec_command('. ~/env/keras/bin/activate && python {}'.format(file))	#runs python script
	for line in stdout.readlines(): print(line)
	for line in stderr.readlines(): print(line)
	ssh_client.close()


#fn.upload_py_files_script('train.py','functions.py')	

#run_file_script('train.py')	

#os.listdir('D://coco_dataset//train2017//')[0]
#fn.upload_py_files_script('000000000009.jpg', where_from='D://coco_dataset//train2017//', where_to='/data/andrej/')

print('ok')
'''
files=[]
for i in range(10):
	files.append('testx{}.npy'.format(i))
	files.append('testy{}.npy'.format(i))
print(files)
#fn.download_files_script(*files, where_to='D://cnn//', where_from='/home/andrej/')

for file in files:
	if file[4]=='x' :
		path = os.path.join(os.path.dirname(__file__),file)
		print(path)
		img = np.load(path)[0]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		cv2.imshow("image", img)
		cv2.waitKey()

'''

'''
#print(subprocess.run(['cd /mnt/d/cnn','mkdir jebem'],shell='C:\\Windows\\System32\\bash.exe'))
#subprocess.run(['D'], shell=True)
#subprocess.run(['cd', 'D:\cnn'], shell=True)

#print(subprocess.check_output(['C:\\Windows\\System32\\bash.exe','-l']))
#,'cd /mnt/d/cnn','ls']))

from subprocess import Popen, PIPE

proc = Popen(['sftp','user@server', 'stop'], stdin=PIPE)
proc.communicate('password')
proc.communicate(input='password')

'''

***************************************************************** VISUALIZATION CODE

# layer.get_weights()[0]
for layer in model.layers: 
	try:
		shape = np.shape(layer.get_weights()[0])
		print(layer.get_config()['name'], shape)#, layer.get_weights())
	except:
		print(layer.get_config()['name'])
#first_layer_weights = model.layers[0].get_weights()[0]
#first_layer_biases  = model.layers[0].get_weights()[1]
#second_layer_weights = model.layers[1].get_weights()[0]
#second_layer_biases  = model.layers[1].get_weights()[1]


ann = models.Sequential()
x = layers.Conv2D(filters=4,kernel_size=(5,3),input_shape=(32,32,2))
ann.add(x)
ann.add(layers.Activation("relu"))
print(np.shape(x.get_weights()[0]))

#for i in range(1):#x_data:
def vizu(i):
	plt.subplot(subplot_rows, subplot_columns, 1)
	#fig, ax = plt.subplots(1)
	mngr = plt.get_current_fig_manager()
	mngr.window.setGeometry(250, 100, 1280, 824)
	print(x_data[1].shape)
	aa = cv2.cvtColor(x_data[i].astype('uint16'), cv2.COLOR_BGR2RGB)
	plt.imshow(aa)
	plt.title('x_data')
	
	plt.subplot(subplot_rows, subplot_columns, 2)
	ab = cv2.cvtColor(y_pred[i].astype('uint16'), cv2.COLOR_BGR2RGB)
	plt.imshow(ab)
	plt.title('y_pred')
	
	plt.subplot(subplot_rows, subplot_columns, 3)
	ac = cv2.cvtColor(y_data[i].astype('uint16'), cv2.COLOR_BGR2RGB)
	plt.imshow(ac)
	plt.title('y_data')
	plt.show()

class Eh():
	def __init__(self, x_data, y_data, y_pred):
		self.x = x_data
		self.y = y_data
		self.yp = y_pred
		self.leng = np.shape(x_data)[0]

	def get_losses(self):
		sess = tf.InteractiveSession()
		#for i in range(self.leng):
		y_true = self.y#[i]
		y_pred = self.yp#[i]
		n=y_true.shape[0]

		no_objects_mask = K.cast(K.equal(y_true, 0),K.floatx())
		objects_mask = K.cast(K.not_equal(y_true, 0),K.floatx())
	
		#no object loss 	
		no_object_loss = K.sum(K.square(0 - (y_pred*no_objects_mask)[:,:,:,0]))
		    
		#object loss 
		object_loss = K.sum(K.square(((1 - y_pred) * objects_mask)[:,:,:,0]))
	
		# loss from bounding boxes
		bb_loss= K.sum(K.square(y_true[:,:,:,1:] - (y_pred * objects_mask)[:,:,:,1:]))
		
		loss= 1 * no_object_loss  +  5 * bb_loss  +  5 * object_loss     	
		print(K.sum(loss).eval()/n)
		
		print((0.5 * no_object_loss).eval()/n, 
			  (5 * bb_loss).eval()/n,
			  (5 * object_loss).eval()/n,
			  loss.eval()/n,'\n')
		
		sess.close()
		
	def mean_squared_error(self):
		sess = tf.InteractiveSession()
		for i in range(self.leng):
			y_true = self.y[i]
			y_pred = self.yp[i]
			print(K.mean(K.square(y_pred - y_true), axis=-1))
		sess.close()	
eh = Eh(xx_data, yy_data, yy_pred)
#eh.get_losses()
#eh.mean_squared_error()

'''
model = Viz.load_model(name = 'best_gen_model.h5',
					   from_server = True,
					   to_return = True,
					   where_from= '/home/andrej/',
					   where_to= 'D://cnn//')

'''



#Loading and downloading model with weights
#fn.download_trained_script(load_name)
#model = fn.load_model('d'+load_name)
#model.compile(optim, loss='mean_squared_error')

#Loading and downloading full model 
#fn.download_files_script(load_name+'.h5', where_to='D://cnn//', where_from='/home/andrej/')
#model = fn.load_model_w_optim(load_name)




***************************************************************** TRAINING CODE


a = [[],[]]
for i in range(len(n['loss'])):
	a[0].append(n['loss'][i][0])
	a[1].append(n['val_loss'][i][0])
n['loss'] = a[0]
n['val_loss'] = a[1]
'''
'''
trained on first dataset, 3 epochs 
best init L = 79   lr=0.0003
my mess2  L = 83   lr=0.0003
trained on first dataset, 1 epochs 	
my mess3  L = 136   lr=0.00003
my mess3  L = 132   lr=0.00008
my mess3  L = 129   lr=0.0003
my mess3  L = 129   lr=0.0008
my mess3  L = 130   lr=0.001



print(K.get_value(model.optimizer.lr))
K.set_value(model.optimizer.lr, 0.001)
print(K.get_value(model.optimizer.lr),'\n','\n','\n','\n')





***************************************************************** DATAGENERATION CODE






plane = np.load('D:\\cnn\\data\\npy_cut_planes_train\\000000551433.npy')
bcg = cv2.imread('D:\\cnn\\data\\all_backgrounds\\000222.png')
bcg = cv2.cvtColor(bcg, cv2.COLOR_BGR2RGB)	
args = [plane, bcg, np.array([0.11277622]), np.array([0.31325853])]
merge_kwargs={'fliph_plane': False,
   'plane_angle': 18,
   'scale': 0.7514797706898554,
   'fliph_bcg': True,
   'bcg_angle': -24,
   'output_shape': (13, 13, 5),
   'ref': (416, 416)}
x_d, y_t = merge(*args, **merge_kwargs)

vizu.plot_x_y(cv2.resize(x_d, (416,416)), y_t)


def merge(plane, bcg, x_pos_r, y_pos_r,\
		  fliph_plane=False, plane_angle=0, scale=1, \
		  fliph_bcg = False, bcg_angle=0, y_true=None, \
		  output_shape=(13,13,5), ref=(416,416), min_pix_plane=10):
	#1.line  = generic arguments
	#2.line  = aircraft specific inputs
	#3.line  = background specific inputs
	#4.line  = most likely not necesarry to specify kwargs
	
	if y_true is None:
		y_true=np.zeros((output_shape))
		no_ac_in=True
	else: no_ac_in=False
	
	if fliph_plane: 		# flip plane horizontally if necesarry
		plane=np.flip(plane, axis=1)
		
	if fliph_bcg and no_ac_in: 			# flip background horizontally if necesarry
		bcg=np.flip(bcg, axis=1)

	if not plane_angle==0:  # rotate plane if specified
		plane = rotate(plane, plane_angle, axes=(1, 0), reshape=True, order=2, mode='constant', cval=0, prefilter=True)
	
		
	if (not bcg_angle==0) and no_ac_in:# and (not no_ac_in):  # rotate plane if specified
		bcg_angle=int(bcg_angle)
		old_sha = bcg.shape		
		#actual background rotation
		rbcg = rotate(bcg, bcg_angle, axes=(1, 0), reshape=True, order=2, mode='constant', cval=0, prefilter=True)		
		bcg_angle=np.abs(bcg_angle)
		if bcg_angle == 45: bcg_angle == 40
		rangle = bcg_angle/57.29578
		new_sha = rbcg.shape
		if not bcg_angle==90:
			# coordinates of middle bottom point after rotation
			midy = np.cos(rangle) * old_sha[0]/2
			midx = np.sin(rangle) * old_sha[0]/2
			#coordinates of where rotated bottom line intersects horizontal(x) and vertical(y) axis with origin in middle point
			xe = midx + midy/np.tan(rangle)
			ye = midy + midx*np.tan(rangle)	
			#black magic which should work for angles -90 to 90 degrees
			if (xe > old_sha[1]) and (ye < old_sha[0]):
				rx = np.cos(rangle)*old_sha[1]/2
				c1 = -(np.tan(90/57.29578-rangle)*rx)
				c2 = -(np.tan(rangle)*xe)
				idkfakt = (np.tan(rangle)-np.tan(90/57.29578-rangle))
				if idkfakt<=0.01: idkfakt=0.01
				x = (c1- c2) / idkfakt		
				new_w = int(2*x)
				new_h = int(-2*(np.tan(rangle)*x + c2))
			elif (xe < old_sha[1]) and (ye < old_sha[0]):
				new_w = int(xe)
				new_h = int(ye)
			elif (xe < old_sha[1]) and (ye > old_sha[0]):
				ry = np.sin(rangle)*old_sha[1]/2
				c1 = -(np.tan(rangle)*ry)
				c2 = -(np.tan(90/57.29578-rangle)*ye)
				idkfakt = (np.tan(rangle)-np.tan(90/57.29578-rangle))
				if idkfakt<=0.01: idkfakt=0.01
				x = -(c1- c2)/ idkfakt
				new_h = int(2*x)
				new_w = int(-2*(np.tan(90/57.29578-rangle)*x + c2))
			#cutting out all invented parts of image
			rbcg=rbcg[int((new_sha[0]/2 - new_h/2)):int((new_sha[0]/2 + new_h/2)), int((new_sha[1]/2 - new_w/2)):int((new_sha[1]/2 + new_w/2)) ,:]
			if not (rbcg.shape[0]<100 or rbcg.shape[1]<100) :
				bcg=rbcg
				
	bcg_sha=bcg.shape
	


	#masking where all pixel values 0   ([0,0,0])
	plane_mask0 = np.ma.masked_not_equal(plane[:,:,0], 0).mask
	plane_mask1 = np.ma.masked_not_equal(plane[:,:,1], 0).mask
	plane_mask2 = np.ma.masked_not_equal(plane[:,:,2], 0).mask	
	plane_mask =  np.ma.masked_equal(plane_mask0, plane_mask1).mask
	plane_mask =  np.ma.masked_equal(plane_mask, plane_mask2).mask
	#cleaning up the edges with dilation and erosion
	n = 2
	for i in range(n): plane_mask=morphology.binary_dilation(plane_mask)
	for i in range(n+3):
		plane_mask=morphology.binary_erosion(plane_mask)

	#cutting off empty space
	rows = np.any(plane_mask, axis=1)
	cols = np.any(plane_mask, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]
	plane_mask = plane_mask[rmin:rmax+1,cmin:cmax+1]
	plane = plane[rmin:rmax+1,cmin:cmax+1,:]
	p_sha = plane.shape
	plane_mask = np.dstack((plane_mask, plane_mask, plane_mask))
	
	#scaling the plane with its mask
	fin_r = scale * min([bcg_sha[0]/p_sha[0], bcg_sha[1]/p_sha[1]]) # scaling with 1 being the background width	
	if fin_r*p_sha[0] * (ref[0]/bcg_sha[0])<min_pix_plane:
		fin_r = fin_r*min_pix_plane/(fin_r*p_sha[0] * (ref[0]/bcg_sha[0]))		
	if fin_r*p_sha[1] * (ref[1]/bcg_sha[1])<min_pix_plane:
		fin_r = fin_r*min_pix_plane/(fin_r*p_sha[1] * (ref[1]/bcg_sha[1]))
	plane = interpolation.zoom(plane, [fin_r ,fin_r , 1], order=4, mode='constant', cval=0)
	plane_mask = interpolation.zoom(plane_mask, [fin_r, fin_r , 1], order=0, mode='constant', cval=False)



	# Setting up values for truth array
	p_sha=plane.shape
	x_pix = int( x_pos_r * (bcg_sha[0] - p_sha[0]) ) # x and y coordinate of top left plane corner in background coordinates
	y_pix = int( y_pos_r * (bcg_sha[1] - p_sha[1]) )
	w_ratio= ref[0] / bcg_sha[0]
	h_ratio= ref[1] / bcg_sha[1]
	x_plane_cen = int( (x_pix+p_sha[0]/2)*w_ratio ) # x and y coordinate of center point for plane in ref coordinates
	y_plane_cen = int( (y_pix+p_sha[1]/2)*h_ratio )
	
	
	box_w= ref[0]/output_shape[0]
	box_h= ref[1]/output_shape[1]
	
	for j in range(output_shape[0]):
		if x_plane_cen>= j*box_w  and  x_plane_cen< (j+1)*box_w:
			a=j
		if y_plane_cen>= j*box_h  and  y_plane_cen< (j+1)*box_h:
			b=j
			
	if (not no_ac_in) and (y_true[a,b,0]==1): # the detector would be responsible for 2 aircraft, so just ignore the new ac
		return bcg, y_true
	else:
		# Filling truth array            
		y_true[a,b,0:]= 1, x_plane_cen, y_plane_cen, p_sha[0]/bcg_sha[0], p_sha[1]/bcg_sha[1]
		y_true[a,b,1]= (y_true[a,b,1]-(a*box_w))/box_w 
		y_true[a,b,2]= (y_true[a,b,2]-(b*box_w))/box_w     
	
		#pasting in aircraft
		bcg[x_pix:x_pix+p_sha[0], y_pix:y_pix+p_sha[1]] = bcg[x_pix:x_pix+p_sha[0], y_pix:y_pix+p_sha[1]] * ~plane_mask  +  plane * plane_mask
		x_data=bcg
		return x_data, y_true
	
	
def gen_training_batch( bcg_dir, plane_dir, n, ref=(416,416), seed=1):
	#setting up directories
	bcg_dir_path = os.path.join(os.path.dirname(__file__),bcg_dir)
	bcg_names = os.listdir(bcg_dir_path)
	plane_dir_path = os.path.join(os.path.dirname(__file__),plane_dir)
	plane_names = os.listdir(plane_dir_path)
	
	random.seed(seed)
	np.random.seed(seed)
	ch_bcg_names = random.sample(bcg_names, n)
	x_data=[]
	y_true=[]
	for name in ch_bcg_names:
		bcg = cv2.imread(os.path.join(bcg_dir_path, name))
		bcg = cv2.cvtColor(bcg, cv2.COLOR_BGR2RGB)	
		
		r_n_planes= np.random.choice(a=[1, 2, 3, 4, 5], p=[150/200, 37/200, 9/200, 3/200, 1/200]) #add 0
		for i in range(r_n_planes):
			plane = np.load(os.path.join(plane_dir_path, random.choice(plane_names)))
			r_unif = np.random.uniform(low=0.0, high=1.0, size=(3,1)) # add min scale
			r_ang_unif = (np.random.uniform(low=-90, high=90, size=(2,1))).astype('int32')
			r_bool = np.random.choice(a=[False, True], size=(5,1), p=[0.5, 0.5])
			if i==0:
				x_d, y_t = merge(plane, bcg, r_unif[0], r_unif[1], fliph_plane=r_bool[0], fliph_bcg=r_bool[1],\
							     plane_angle=r_ang_unif[0], bcg_angle=r_ang_unif[1], scale=(r_unif[2]/2+0.02).item(), ref=ref)
			elif i>0:
				x_d, y_t = merge(plane, x_d, r_unif[0], r_unif[1], fliph_plane=r_bool[0], fliph_bcg=r_bool[1], \
							     plane_angle=r_ang_unif[0], bcg_angle=r_ang_unif[1], scale=(r_unif[2]/2+0.02).item(), y_true=y_t, ref=ref)

		#appending lists
		x_d = cv2.resize(x_d, ref)
		x_data.append(x_d)
		y_true.append(y_t)
		
	x_data = np.vstack( [x_data] )
	y_true = np.vstack( [y_true] )
	x_data = (x_data / 255)
	return x_data, y_true


	#x_data, y_true = gen_training_batch( bcg_dir, plane_dir, 15)
	
	
	for i in x_data:
		fig, ax = plt.subplots(1)
		mngr = plt.get_current_fig_manager()
		mngr.window.setGeometry(250, 100, 1280, 824)
		ax.imshow(i)
		plt.show()



******************************************************************** Tester code

with open('ground_truth_boxes.json') as infile:
	gt_boxes = json.load(infile)

with open('predicted_boxes.json') as infile:
	pred_boxes = json.load(infile)


import h5py
layers_dict = h5py.File('D:\\cnn\\weights\\keras_weights.h5', 'r')
print(list(layers_dict.keys()))
layers_dict['input_2']
'''
'''
from final_viz import Visualization

Viz = Visualization(model= None,
					optim = 'Nadam',
					loss_function = fn.custom_loss,
					ref_x=(416,416,3),
					ref_y=(13,13,5),
					threshold= 1e-4,
					plot_window=[250, 100, 1280, 824])

yolomodel = Viz.load_model(name = 'best_tiny_yolo.h5',
			   from_server = False,
			   to_return = True,
			   where_from= 'D://cnn//weights//',
			   where_to= '')

makemodel = fn.my_models(optimizer='Nadam', loss=fn.custom_loss)
#mymodel = makemodel.yolo_arch(summary=True)
mymodel = makemodel.yolo_arch(summary=False)
#yolomodel.summary()


bcg_names, plane_names = get_bcg_plane_names()

train_params = {'bcg_paths': bcg_names[:int(len(bcg_names)*0.05)],
			    'plane_paths': plane_names[:int(len(bcg_names)*0.05)],
				'steps_per_epoch': 5,
				'batch_size': 16,
				'ref_in': (416,416,3),
				'ref_out': (13,13,5)}			

gen_data(train_params, 20, save_paths=['gen_val_x.npy','gen_val_y.npy'])



xx_data = np.load('gen_train_x.npy').astype('float32')/255
yy_data = np.load('gen_train_y.npy')
print(yy_data.dtype)


print('{0:3} {1:30}   {2}'.format('N', 'My model', 'Yolo model'))
#num_layers = np.max(len(mymodel.layers),len(yolomodel.layers))
for i in range(np.max([len(mymodel.layers),len(yolomodel.layers)])):
	try: 
		mylayer = mymodel.layers[i-1]
		try:
			shape = np.shape(mylayer.get_weights()[0])
			a = mylayer.get_config()['name'] + str(shape)	
		except:
			a = mylayer.get_config()['name']
	except: a = ''
	try: 
		yololayer = yolomodel.layers[i]
		try:
			shape = np.shape(yololayer.get_weights()[0])
			b = yololayer.get_config()['name'] + str(shape)	
		except:
			b = yololayer.get_config()['name']
	except: b = ''	
	print('{0:3} {1:30}   {2}'.format(i, a, b))
	#print(f'{a}{b:>35}','\n')
	

x = mymodel.layers[0]
filters = 16
subplot_rows=1
subplot_columns=16

input_z = np.shape(x.get_weights()[0])[2]
filt_dims = np.shape(x.get_weights()[0])
plt.figure(0)
# subplot(2,3,4) 2 rows,  3 columns,  4th element(2nd row 1st column)
x1w = x.get_weights()[0][:,:,:,:]
n=1
for n_filt in range(1,filters+1):
	plt.subplot(subplot_rows, subplot_columns, n)
	n+=1
	z_flattened = x1w[:,:,0,n_filt-1]
	for z_layer in range(input_z-1):
		z_flattened =  np.concatenate((z_flattened, x1w[:, :, z_layer+1, n_filt-1]), axis=0)
	plt.imshow(z_flattened, interpolation="nearest", cmap="coolwarm") #nearest, bicubic
plt.show()

	
n = 0
i = 0
stop = 12
while n < stop:
	i+=1
	mylayer = mymodel.layers[i-1]
	yololayer = yolomodel.layers[i]
	print('{0:25}  i={1}  n={2}'.format(mylayer.get_config()['name'],i,n))
	try: 
		shape = np.shape(mylayer.get_weights()[0])
		#print(shape)
		#print(yololayer.get_weights())
		#print(mylayer.get_weights())
		try:
			weights = yololayer.get_weights()
			mylayer.set_weights(weights)
			print(mylayer.get_config()['name'], 'weigths set normally  ')
		except:
			try:				
				mylayer.set_weights([yololayer.get_weights()[0], np.zeros((shape[-1],), dtype='float32')])
				print(mylayer.get_config()['name'], 'weigths set with added biases  ')
			except:print(mylayer.get_config()['name'], 'weights not set ')
				
		#print(mylayer.get_weights())
		n+=1	
	except Exception as e:
		pass#print('Error: ',e)


fn.save_model(mymodel, 'best_initiate')
#fn.save_model_w_optim(mymodel, 'best_initiate')
#weights = old_model_layer.get_weights()
#new_model_layer.set_weights(weights)
'''



'''

ref=(416,416,3)

a = np.random.uniform(low=0, high=255, size=(416,416,3)).astype('uint8')

size = 4
aa = np.random.uniform(low=0, high=255, size=(size,size,3)).astype('uint8')
print(np.shape(aa))
aab = interpolation.zoom(aa, [ref[0]/size ,ref[1]/size , 1], order=0, mode='constant', cval=0)
print(np.shape(aab),'aaaaa')
sigma = 30
for i in range(3) :aab[:,:,i] = gaussian_filter(aab[:,:,i] , sigma=sigma)

aab = gaussian_filter(aab , sigma=0)

diff = np.max(aab) - np.min(aab)
aab = aab * 255./diff
print( np.max(aab))
aab = aab - np.min(aab)
print( np.max(aab))
aab = aab.astype('uint8')


subplot_rows = 1
subplot_columns = 2


plt.subplot(subplot_rows, subplot_columns, 1)
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(250, 100, 1280, 824)
plt.imshow(a)
plt.subplot(subplot_rows, subplot_columns, 2)
plt.imshow(aab)
plt.show()


print(np.shape(aab))
aab =  np.stack((aab, aab))#, axis=3)
print(np.shape(aab))


a= np.zeros((15000, 416, 416, 3), dtype='uint8')
print( a.nbytes,  a.nbytes/1000/1000/1000)

chunks_size=5
aala= np.linspace(0,29,30)
parts = int(len(aala) / chunks_size)
if not len(aala)%chunks_size==0 : parts+=1
n=0

for part in range(parts):
	x_data=np.zeros((chunks_size,416, 416, 3), dtype='uint8')
	y_data=np.zeros((chunks_size,416, 416, 3), dtype='uint8')
	print('oke')
	n-=1
	for pic_name in range(len(aala)):
		n+=1
		if (stop and n>chunks_size*(part+1)) or n>(len(aala)-1) : break			

		print(aala[n], part)




import script

print('oke')




************************************************************* OLD FUNCTIONS







def custom_loss(y_true, y_pred, no_object_scale=0.5, bb_scale=5., object_scale=5.): #(0.5,5,5)
    #shape = y_true.shape #(13,13,5)
    
    #no object loss 
	#no_objects_mask = ma.masked_equal(y_true, 0).mask
	no_objects_mask = K.cast(K.equal(y_true, 0),K.floatx())
	no_object_loss = K.sum(K.square(0 - (y_pred*no_objects_mask)[:,:,0]), axis=-1)
	no_object_loss = tf.cast(no_object_loss, tf.float32)
	    
	#object loss 
	object_loss = K.sum(K.square(((1 - y_pred) * ~no_objects_mask)[:,:,0]), axis=-1)
	object_loss = tf.cast(object_loss, tf.float32)
    
    # loss from bounding boxes
	bb_loss= K.sum(K.square(y_true[:,:,1:] - (y_pred * ~no_objects_mask)[:,:,1:]), axis=-1)
	bb_loss = tf.cast(bb_loss, tf.float32)

	loss= no_object_scale * no_object_loss  +  bb_scale * bb_loss  +  object_loss*object_scale
    #K.print(loss)      
	return 0*loss

def custom_loss(y_true, y_pred):#, no_object_scale=0.5, bb_scale=5., object_scale=5.): #(0.5,5,5)
    #shape = y_true.shape #(13,13,5)
    
	no_objects_mask = K.cast(K.equal(y_true, 0),K.floatx())
	objects_mask = K.cast(K.not_equal(y_true, 0),K.floatx())
	
	no_object_loss = K.square(0 - (y_pred*no_objects_mask)[:,:,0])
	    
	object_loss = K.square(((1 - y_pred) * objects_mask)[:,:,0])
    
	bb_loss= K.sum(K.square(y_true[:,:,1:] - (y_pred * objects_mask)[:,:,1:]), axis=-1)

	loss= 0.5 * no_object_loss  +  5 * bb_loss  +  5 * object_loss	    
	return loss


no_object_scale=0.5
bb_scale=5.
object_scale=5.

from keras import backend as K
import numpy.ma as ma

object_scale = np.array([5.]).astype('float32')
bb_scale = np.array([5.]).astype('float32')
no_object_scale = np.array([0.5]).astype('float32')
K.dtypes.cast(bb_scale, 'float32', name=None)

no_object_loss
object_loss
bb_loss

import tensorflow as tf
sess = tf.InteractiveSession()
custom_loss(y_true, y_pred, no_object_scale=0.5, bb_scale=5., object_scale=5.):

bb_loss = tf.cast(bb_loss, tf.float32)

sess = tf.InteractiveSession()
(no_object_scale * no_object_loss).eval()
(bb_scale * bb_loss).eval()
(object_loss*object_scale).eval()




	def my_mess(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(16, 3, strides=(1, 1), padding='same',input_shape=(416, 416, 3)))
		model.add(layers.LeakyReLU(alpha=0.1))
		
		filters=[32,64,128,256,512]#,1024]
		for fil in filters:
		    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		    model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same')) 
		    model.add(layers.LeakyReLU(alpha=0.1))
		
		model.add(layers.Conv2D(512, 3, strides=(1, 1), padding='same'))
		model.add(layers.Conv2D(256, 1, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(64, 5, strides=(1, 1), padding='same'))  
		model.add(layers.Conv2D(5, 5, strides=(1, 1), padding='same'))  
		#model.compile('Nadam', loss=custom_loss, sample_weight_mode = None)#  'mean_squared_error'
		#if summary: model.summary()


		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model


	def my_mess2(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(32, 3, strides=(1, 1), padding='same',input_shape=(416, 416, 3)))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		
		model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(32, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		
		model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))		
		
		model.add(layers.Conv2D(256, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(256, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))	
		
		model.add(layers.Conv2D(512, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))	
		model.add(layers.Conv2D(256, 1, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(128, 5, strides=(1, 1), padding='same'))  
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(64, 5, strides=(1, 1), padding='same'))  
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(5, 5, strides=(1, 1), padding='same'))  
		model.compile('Nadam', loss=custom_loss, sample_weight_mode = None)#  'mean_squared_error'
		model.summary()


		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model

	def my_mess3(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(16, 3, strides=(1, 1), padding='same',input_shape=(416, 416, 3)))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		
		model.add(layers.Conv2D(32, 3, strides=(1, 1), padding='same',input_shape=(416, 416, 3)))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		
		model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		
		model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))		
		
		model.add(layers.Conv2D(256, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))	
		
		model.add(layers.Conv2D(64, 1, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(16, 3, strides=(1, 1), padding='same'))  
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Flatten(data_format=None))
		model.add(layers.Dense(845))
		model.add(layers.Reshape((13,13,5)))	
		model.compile('Nadam', loss=custom_loss, sample_weight_mode = None)#  'mean_squared_error'
		model.summary()


		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model

	def my_mess4(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(16, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4), input_shape=(416, 416, 3)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		
		filters=[32,64,128,256,512]#,1024]
		for fil in filters:
		    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		    model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
		    model.add(layers.BatchNormalization())
		    model.add(layers.LeakyReLU(alpha=0.1))

		#model.add(layers.Conv2D(1024, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
		#model.add(layers.BatchNormalization())
		#model.add(layers.LeakyReLU(alpha=0.1))
		
		model.add(layers.Conv2D(256, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(5, 3, strides=(1, 1), padding='same', activation='linear', kernel_regularizer= l2(5e-4)))  
		model.add(layers.Flatten(data_format=None))
		model.add(layers.Dense(845))
		model.add(layers.Reshape((13,13,5)))
		model.compile('Nadam', loss=custom_loss, sample_weight_mode = None)#  'mean_squared_error'
		#model.summary()
		
		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model


	def my_mess5(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(16, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4), input_shape=(416, 416, 3)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		
		filters=[32,64,128,256]#,512]#,1024]
		for fil in filters:
		    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		    model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
		    model.add(layers.BatchNormalization())
		    model.add(layers.LeakyReLU(alpha=0.1))

		#model.add(layers.Conv2D(1024, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
		#model.add(layers.BatchNormalization())
		#model.add(layers.LeakyReLU(alpha=0.1))
		
		model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(5, 1, strides=(1, 1), padding='same', activation='linear', kernel_regularizer= l2(5e-4)))  
		model.add(layers.Flatten(data_format=None))
		model.add(layers.Dense(845))
		model.add(layers.Reshape((13,13,5)))
		#model.compile('Nadam', loss=custom_loss, sample_weight_mode = None)#  'mean_squared_error'
		#model.summary()
		
		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model

	def yolo_arch_v3(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(32, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		
		filters=[64,128,256,512,1024]
		for fil in filters:
			model.add(layers.Conv2D(int(fil/2), 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
			model.add(layers.BatchNormalization())
			model.add(layers.LeakyReLU(alpha=0.1))
			model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
			model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
			model.add(layers.BatchNormalization())
			model.add(layers.LeakyReLU(alpha=0.1))

		
		model.add(layers.Conv2D(512, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(5, 3, strides=(1, 1), padding='same', activation='linear', kernel_regularizer= l2(5e-4)))  
		model.add(layers.Activation('sigmoid'))
		
		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model

	def yolo_arch_v2(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(32, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		
		filters=[64,128,256,512,1024]
		for fil in filters:
		    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		    model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
		    model.add(layers.BatchNormalization())
		    model.add(layers.LeakyReLU(alpha=0.1))

		
		model.add(layers.Conv2D(512, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(5, 3, strides=(1, 1), padding='same', activation='linear', kernel_regularizer= l2(5e-4)))  
		#model.add(layers.Activation('sigmoid'))
		model.add(layers.ReLU(max_value= 1, negative_slope= 0.0, threshold= 0.0))
		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model

	def yolo_arch(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(16, 3, strides=(1, 1), padding='same',input_shape=(416, 416, 3)))
		#model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		
		filters=[32,64,128,256,512]#,1024]
		for fil in filters:
		    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		    model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same')) 
		    #model.add(layers.BatchNormalization())
		    model.add(layers.LeakyReLU(alpha=0.1))
		
		model.add(layers.Conv2D(512, 3, strides=(1, 1), padding='same'))
		#model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(5, 3, strides=(1, 1), padding='same'))  
		#model.add(layers.Activation('sigmoid'))
		#model.add(layers.ReLU(max_value= 1, negative_slope= 0.0, threshold= 0.0))
		model.compile('Nadam', loss=custom_loss, sample_weight_mode = None)
		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model

	def yolo_arch_full(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(16, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4), input_shape=(416, 416, 3)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		
		filters=[32,64,128,256,512]#,1024]
		for fil in filters:
		    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		    model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
		    model.add(layers.BatchNormalization())
		    model.add(layers.LeakyReLU(alpha=0.1))

		model.add(layers.Conv2D(1024, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		
		model.add(layers.Conv2D(512, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(5, 3, strides=(1, 1), padding='same', activation='linear', kernel_regularizer= l2(5e-4)))  
		#model.add(layers.Activation('sigmoid'))
		#model.add(layers.ReLU(max_value= 1, negative_slope= 0.0, threshold= 0.0))
		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model




***************************************************************** PLAYGROUND ZERO






import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras import layers
import keras
import keras.backend as K
import os
import functions as fn
keras.backend.clear_session()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

dirname = os.path.dirname(os.path.abspath(__file__))
data_dir= os.path.join(dirname, 'data')
bcg_dir = os.path.join(data_dir, 'all_backgrounds')			# dir for all empty backgrounds
plane_dir = os.path.join(data_dir, 'npy_cut_planes_train')	# npy file for arrays of cut out airplanes

np.random.seed(1)
bcg_names = os.listdir(bcg_dir)
bcg_names = [os.path.join(bcg_dir, x) for x in bcg_names]
np.random.shuffle(bcg_names)
plane_names = os.listdir(plane_dir)
plane_names = [os.path.join(plane_dir, x) for x in plane_names]
np.random.shuffle(plane_names)
eval_params = {'bcg_paths': bcg_names[:int(len(bcg_names)*0.05)],
			    'plane_paths': plane_names[:int(len(bcg_names)*0.05)],
				'steps_per_epoch': 5,
				'batch_size': 10,
				'ref_in': (416,416,3),
				'ref_out': (13,13,5)}

eval_generator = fn.DataGenerator( **eval_params)
'''
train_images = mnist.train_images() 
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)
'''

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

num_filters = 8
filter_size = 3
pool_size = 2
'''
# Build the model.
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

'''
model = Sequential()
model.add(layers.Conv2D(8, 3, strides=(1, 1), padding='same', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.1))
filters=[64,128,256,512,1024]
for i in range(0):
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='same')) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D(pool_size=(pool_size,pool_size), strides=(2,2), padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense((10)))#,activation='softmax'))
#model.add(layers.ReLU(max_value= 1, negative_slope= 0.0, threshold= 0.0))
model.summary()
#model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'

'''
		model = models.Sequential()
		model.add(layers.Conv2D(32, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		
		filters=[64,128,256,512,1024]
		for fil in filters:
		    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		    model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4))) 
		    model.add(layers.BatchNormalization())
		    model.add(layers.LeakyReLU(alpha=0.1))

		
		model.add(layers.Conv2D(512, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(5, 3, strides=(1, 1), padding='same', activation='linear', kernel_regularizer= l2(5e-4)))  
		#model.add(layers.Activation('sigmoid'))
		model.add(layers.ReLU(max_value= 1, negative_slope= 0.0, threshold= 0.0))
		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'

'''


# Compile the model.
optim = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
'''
def custom_loss(y_true, y_pred):
	diff = K.square(y_pred - y_true)
	return diff

def custom_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def custom_loss(y_true, y_pred):#, no_object_scale=0.5, bb_scale=5., object_scale=5.): #(0.5,5,5)
    #shape = y_true.shape #(13,13,5)
    
	no_objects_mask = K.cast(K.equal(y_true, 0),K.floatx())
	objects_mask = K.cast(K.not_equal(y_true, 0),K.floatx())
	
	no_object_loss = K.square(0 - (y_pred*no_objects_mask)[:,:,0])
	    
	object_loss = K.square(((1 - y_pred) * objects_mask)[:,:,0])
    
	bb_loss= K.sum(K.square(y_true[:,:,1:] - (y_pred * objects_mask)[:,:,1:]), axis=-1)

	loss= 0.5 * no_object_loss  +  5 * bb_loss  +  5 * object_loss	    
	return loss

'''
def custom_loss(y_true, y_pred):
	#K.reshape(y_pred, (5,2))
	diff = K.square(y_pred[:,0:5] - y_true[:,0:5])
	diffo = K.square(y_pred[:,5:10] - y_true[:,5:10])
	return K.sum(diff+diffo)





model.compile(
  optim,
  loss=custom_loss,
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=1,
  validation_data=(test_images, to_categorical(test_labels)),
)

# Save the model to disk.
#model.save_weights('cnn.h5')

# Load the model from disk later using:
# model.load_weights('cnn.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:10])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:10]) # [7, 2, 1, 0, 4]








***********************************************************************   PRETRAINING

import functions as fn
start = fn.time_print(None, None, None)
import matplotlib.pyplot as plt
import numpy as np
#from keras import backend as K
#from keras.models import Model
from keras.regularizers import l2
from keras import models, callbacks
from keras import layers
import os
import platform
import json, codecs
import gc
#from sys import getsizeof
#from tqdm import tqdm
import warnings
import adjusted_callbacks as fwc
from keras import backend as K
import keras
import tensorflow as tf
import time




def modelv1(optim):
	model = models.Sequential()
	model.add(layers.Conv2D(32, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=0.1))
	
	model.add(layers.Conv2DTranspose(3, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
	#model.add(layers.BatchNormalization())
	#model.add(layers.LeakyReLU(alpha=0.1))	
	
	model.compile(optim, loss='mean_squared_error')
	model.summary()
	return model


def modelv2(optim):
	model = models.Sequential()
	model.add(layers.Conv2D(32, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=0.1))
	model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
	
	model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=0.1))	
	
	model.add(layers.Conv2DTranspose(32, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=0.1))
	model.add(layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
	
	model.add(layers.Conv2DTranspose(3, 3, strides=(1, 1), padding='same', kernel_regularizer= l2(5e-4),input_shape=(416, 416, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=0.1))
	
	model.compile(optim, loss='mean_squared_error')
	model.summary()
	return model
	
	
def pretraining():
	start = fn.time_print(None, None, None)
	warnings.filterwarnings('ignore', '.*output shape of zoom.*')
	
	stop = fn.time_print(start, None, 'Modules imported')
	
	'''
	greedy layer-wise pretraining
	unpooling vs upsampling
	conv vs convT
	https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation/blob/master/DeconvNet.py
	https://keras.io/layers/convolutional/
	https://blog.keras.io/building-autoencoders-in-keras.html
	https://stackoverflow.com/questions/43305891/how-to-correctly-get-layer-weights-from-conv2d-in-keras
	'''
	
	if platform.system() == 'Windows':
		print('running on Windows')
		'''
		to_save_model = True
		to_load_model = True
		train_model = True
		'''
		to_save_model = False
		to_load_model = True
		train_model = True	
		
		dirname = os.path.dirname(os.path.abspath(__file__))
		save_dir = dirname
		data_dir= os.path.join(dirname.split('\\')[0], '\\coco_dataset')
		data_dir= os.path.join(data_dir, 'train2017')
		
	else: 
		print('running on {}'.format(platform.system()))
		to_save_model = True
		to_load_model = True
		train_model = True	
		history_filename = '/home/andrej/history.json'
		data_dir = '/data/andrej'
		save_dir = os.path.abspath(".")#'/data/andrej/'
		save_dir = os.path.join(save_dir, 'pretrain_data')
		save_dir = '/data/andrej/'
		print(save_dir)
	
	
	save_name = 'prev2'
	load_name = save_name
	optim='Nadam'
	
	

	
	
	class LossHistory(callbacks.Callback):	
	    # https://stackoverflow.com/a/53653154/852795
	    def on_epoch_end(self, epoch, logs = None):
	        new_history = {}
	        for k, v in logs.items(): # compile new history from logs
	            new_history[k] = [v] # convert values into lists
	        current_history = loadHist(history_filename) # load history from current training
	        current_history = appendHist(current_history, new_history) # append the logs
	        saveHist(history_filename, current_history) # save history from current training
	
	
	def saveHist(path, history):
	    with codecs.open(path, 'w', encoding='utf-8') as f:
	        json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4) 
	
	def loadHist(path):
	    n = {} # set history to empty
	    if os.path.exists(path): # reload history if it exists
	        with codecs.open(path, 'r', encoding='utf-8') as f:
	            n = json.loads(f.read())
	    return n
	
	def appendHist(h1, h2):
	    if h1 == {}:
	        return h2
	    else:
	        dest = {}
	        for key, value in h1.items():
	            dest[key] = value + h2[key]
	        return dest
	



	
	
	if to_load_model: 	# load json and create model
		model = fn.load_model(load_name)
		model.compile(optim, loss='mean_squared_error')
		stop = fn.time_print(start, stop, 'Loaded model from disk')
	
	else: 			# Create model architecture from scratch
		model = modelv2(optim)
		stop = fn.time_print(start, stop, 'Model compiled')
	
	
	if train_model:	 #loading training data and training on it
		files = os.listdir(data_dir)
		n=0
		contin = True
		history = {}
		#for file in files:	print(file)
		#x_data, y_data = fn.prep_pretrain_pics(data_dir=data_dir, save_dir=save_dir, ref=(416,416), stop=True, \
		#									   startval=2500, stopval=6500, save=True, load=True)
		
		# pretrain_y0part.npy
		while contin:
			contin = False
			for file in files:	
				train = False
				if file.split('_')[0]=='pretrain':
					digits = [s for s in file if s.isdigit()]
					num=''
					number = num.join(s for s in digits)
					if number==str(n) and file.split('_')[1][0]=='x' :
						train = True
						contin = True
						n+=1
						
						max_index=2000
						
						try:
							x_data=np.load(os.path.join(data_dir,file))
							y_data=np.load(os.path.join(data_dir,file.split('.')[0].replace('y','x')+'.npy'))
							max_index=None#2000
							if n==1:
								val_x, x_data = x_data[0:1000], x_data[1000:max_index]
								val_y, y_data = y_data[0:1000], y_data[1000:max_index]
							else:
								x_data = x_data[0:max_index]
								y_data = y_data[0:max_index]
						except Exception as e:
							print('Error: ',e)
							contin = False
					
					if train:
						print('\n training on ',n)
						gc.collect()
						history_cp = LossHistory()
						if not history=={}:
							last_best = min(history['val_loss'])
							wait = len(history['val_loss']) - history['val_loss'].index(last_best) - 1
						else:
							last_best = np.Inf
							wait = 0
						my_checkpoint = fwc.ModelCheckpoint('bbest_'+save_name+'.h5', monitor='val_loss',\
														    mode='min', save_best_only=True,previous_best=last_best, \
															verbose=1)
						es = fwc.EarlyStopping(monitor='val_loss', mode='min',wait=wait, baseline=last_best, \
										       verbose=1, patience=6)
						NAME = 'pretraining_{}'.format(int(time.time()))
						TB = callbacks.TensorBoard(log_dir='./logs/{}'.format(NAME))
						#cb = callbacks.ModelCheckpoint('best_'+save_name+'.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
						cb_list = [es, my_checkpoint, history_cp,TB]					
						
						#with tf.Session( config = tf.ConfigProto( log_device_placement = True ) ):
						new_history = model.fit(x=x_data, y=y_data, batch_size=16, epochs=4, verbose=1, \
										           validation_data=(val_x, val_y), shuffle=False, callbacks=cb_list)
						

						history = appendHist(history, new_history.history)
						
						stop = fn.time_print(start, stop, 'model trained on pretraining data')
						#plt.plot(history['loss'], label='train')
						#plt.plot(history['val_loss'], label='test')
						#plt.legend()
						#plt.show()
						if n>=2:
							print(history)
							quit()
						if to_save_model:
							fn.save_model(model, save_name)
							stop = fn.time_print(start, stop, 'Saved model {} to disk'.format(save_name))
		
		print(history)		
						


if __name__ == '__main__':
	'''
	config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} ) 
	sess = tf.Session(config=config) 
	keras.backend.set_session(sess)
	print('aaaaaaaa')
	
	from tensorflow.python.client import device_lib
	
	print(device_lib.list_local_devices())
	print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
	K.tensorflow_backend._get_available_gpus()
	'''
	pretraining()


#print('T+'+str(int((time.process_time() - start) * 100) / 100.),'sec -- program finished')

'''
Epoch 00002: val_loss improved from 6631.95591 to 6384.55934, saving model to best_prev2.h5
Epoch 3/4
7500/7500 [==============================] - 556s 74ms/step - loss: 6550.8776 - val_loss: 6282.8078

Epoch 00003: val_loss improved from 6384.55934 to 6282.80776, saving model to best_prev2.h5


'''






********************************************************************* SOME PREP

import functions as fn
import numpy as np
import os

dirname = os.path.dirname(os.path.abspath(__file__))
data_dir= os.path.join(dirname, 'data')

train_pics_dir=os.path.join(data_dir, 'train_pics')
train_annot_dir=os.path.join(data_dir, 'train_annot')

eval_pics_dir=os.path.join(data_dir, 'eval_pics')
eval_annot_dir=os.path.join(data_dir, 'eval_annot')


if False: anchors, X=(fn.anchors_from_data(4, train_pics_dir, train_annot_dir))

x_train_data, y_train_data=fn.prep_pics( img_folder=train_pics_dir,  annot_dir=train_annot_dir,  stop=True, stopval=100)
np.save(os.path.join(data_dir,'x_train_data'),x_train_data)
np.save(os.path.join(data_dir,'y_train_data'),y_train_data)


x_eval_data, y_eval_data=fn.prep_pics( img_folder=eval_pics_dir,  annot_dir=eval_annot_dir,  stop=False)
np.save(os.path.join(data_dir,'x_eval_data'),x_eval_data)
np.save(os.path.join(data_dir,'y_eval_data'),y_eval_data)


	if False: anchors, X=(fn.anchors_from_data(4, train_pics_dir, train_annot_dir))
	
	x_train_data, y_train_data=fn.prep_pics(names='good_pic_names.npy', data_dir=data_dir, img_folder=train_pics_dir, \
											annot_dir=train_annot_dir,  stop=False, stopval=100)
	np.save(os.path.join(data_dir,'x_train_data'),x_train_data)
	np.save(os.path.join(data_dir,'y_train_data'),y_train_data)
	
	'''
	x_eval_data, y_eval_data=fn.prep_pics(names='good_pic_names.npy', data_dir=data_dir, img_folder=eval_pics_dir,  \
										  annot_dir=eval_annot_dir,  stop=False)
	np.save(os.path.join(data_dir,'x_eval_data'),x_eval_data)
	np.save(os.path.join(data_dir,'y_eval_data'),y_eval_data)
	'''






************************************************************************* OLD TRAIN

import time
import functions as fn
import numpy as np
#from keras import backend as K
#from keras.models import Model
from keras.regularizers import l2
from keras import models, callbacks
from keras import layers
import os
from sys import getsizeof
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
import adjusted_callbacks as fwc
import json, codecs




def start_training(to_save=False, to_load = False, to_train_onOG = False, to_train_onGEN = False):
	start = fn.time_print(None, None, None)
	stop = fn.time_print(start, None, 'Modules imported')

	gen_batch = 64*15	 	    # amount in a batch for auto generated pics
	gen_epochs = 5		# number of batch for auto generated pics
	save_name = 'test_model'
	history_filename = '/home/andrej/history.json'
	load_name = save_name
	optim='Nadam'
		
	dirname = os.path.dirname(os.path.abspath(__file__))
	data_dir= os.path.join(dirname, 'data')
	bcg_dir = os.path.join(data_dir, 'all_backgrounds')			# dir for all empty backgrounds
	plane_dir = os.path.join(data_dir, 'npy_cut_planes_train')	# npy file for arrays of cut out airplanes 
	
	
	
	if to_load: 	# load json and create model
		model = fn.load_model(load_name)
		model.compile(optim, loss=fn.custom_loss)
		stop = fn.time_print(start, stop, 'Loaded model from disk')
	
	else: 			# Create model architecture from scratch
		model = create_cnn(start, stop, optim)
	
	if to_train_onOG:	 #loading original training data and training on it
		x_train_data=np.load(os.path.join(data_dir,'x_train_data.npy'))
		y_train_data=np.load(os.path.join(data_dir,'y_train_data.npy'))
		model.fit(x=x_train_data, y=y_train_data, batch_size=30, epochs=20, verbose=1, validation_split=0.2, shuffle=True)
		stop = fn.time_print(start, stop, 'model trained on OG data')
		
	if to_train_onGEN: 	#training on automatically generated data
		history = {}
		for i in range(gen_epochs):
			if ((i+1) % 1) == 0:
				fn.save_model(model, save_name)		
				print('model saved')
			print('Epoch {}/{}'.format(i+1,gen_epochs))
			trseed = int(np.random.uniform(low=0, high=100000))
			x_data, y_true = fn.gen_training_batch( bcg_dir, plane_dir, gen_batch, seed=trseed)
			print(getsizeof(x_data),getsizeof(y_true))
			#model.fit(x=x_data, y=y_true, batch_size=int(gen_batch*0.8), epochs=5, verbose=1, validation_split=0.2, shuffle=True)
			
			history_cp = LossHistory()
			if not history=={}:
				last_best = min(history['val_loss'])
				wait = len(history['val_loss']) - history['val_loss'].index(last_best) - 1
			else:
				last_best = np.Inf
				wait = 0
			my_checkpoint = fwc.ModelCheckpoint('bbest_'+save_name+'.h5', monitor='val_loss',\
											    mode='min', save_best_only=True,previous_best=last_best,verbose=1)
												
			es = fwc.EarlyStopping(monitor='val_loss', mode='min',wait=wait, baseline=last_best,verbose=1, patience=6)
			#cb = callbacks.ModelCheckpoint('best_'+save_name+'.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
			cb_list = [es, my_checkpoint, history_cp]					
			
			
			
			new_history = model.fit(x=x_data, y=y_true, batch_size=64, epochs=10, verbose=1, validation_split=0.2, shuffle=True, callbacks=cb_list)
			history = appendHist(history, new_history.history)

		t_per_pic=int((time.process_time() - stop)/gen_batch/gen_epochs * 100) / 100.
		print(t_per_pic, t_per_pic*100000/3600,'h')
		stop = fn.time_print(start, stop, 'model trained on generated data (batch: {}, epochs: {})'.format(gen_batch, gen_epochs))
		print(history)		
		
	if to_save:
		fn.save_model(model, save_name)
		stop = fn.time_print(start, stop, 'Saved model to disk')
	print('T+'+str(int((time.process_time() - start) * 100) / 100.),'sec -- start_training finished')


class LossHistory(callbacks.Callback):	
    # https://stackoverflow.com/a/53653154/852795
    def on_epoch_end(self, epoch, logs = None):
        new_history = {}
        for k, v in logs.items(): # compile new history from logs
            new_history[k] = [v] # convert values into lists
        current_history = loadHist(history_filename) # load history from current training
        current_history = appendHist(current_history, new_history) # append the logs
        saveHist(history_filename, current_history) # save history from current training

if __name__ == '__main__':
	to_save = False
	to_load = False
	to_train_onOG = False
	to_train_onGEN = False
	
	#to_save = True
	#to_load = True
	#to_train_onOG = False
	#to_train_onGEN = True	
	start_training(to_save=to_save, to_load = to_load, to_train_onOG = to_train_onOG, to_train_onGEN = to_train_onGEN)








************************************************************* FIRST PYTHON VERSION AFTER JUPYTER




from sklearn.cluster import KMeans
import cv2             # working with, mainly resizing, images
from tqdm import tqdm
from lxml import etree
import os                  # dealing with directories
import numpy as np
import numpy.ma as ma

from keras import backend as K
from keras.regularizers import l2
from keras import models
from keras import layers


TRAIN_annot_dir = 'D:/cnntry/air_data/annot_train'
img_folder = 'D:/cnntry/air_data/trainairplane'

save_path='D:/cnntry'
DATA_file= 'train_airdata.npy' 
num_anchors= 3

input_shape = (416,416)
h, w = input_shape


def read_xml(img_name):
    annot_label = img_name.split('.')[0]+'.xml'           #xml file name
    annot_xml = os.path.join(TRAIN_annot_dir,annot_label)  #xml file full path
    root = etree.parse(annot_xml).getroot()
    #for child in root: print(child.tag, child.attrib)        
    img_w= int(root[3][0].text)
    img_h= int(root[3][1].text)
    bb=[]                                   #list for bounding boxes
    for plane in root.iter("object"):
        for l in range(4): bb.append(int(plane[4][l].text)) 
    return img_w, img_h, bb

def rescale_bb(img_w, img_h, bb, ref=(416,416)):
    w_ratio= ref[0] / img_w
    h_ratio= ref[1] / img_h
    for i in range(0,len(bb),2):
        bb[i]= int(bb[i]* w_ratio)
        bb[i+1]= int(bb[i+1]* h_ratio)
    return bb

def conv_bb_coord(bb):
    for i in range(0,len(bb),4):
        x_cen= int( (bb[i]+ bb[i+2])/2)
        y_cen= int( (bb[i+1]+ bb[i+3])/2)
        w= bb[i+2]- bb[i]
        h= bb[i+3]- bb[i+1]
        bb[i],bb[i+1],bb[i+2],bb[i+3]= x_cen, y_cen, w, h
    return bb


def create_truth_arr(bb, output_shape=(13,13,5), ref=(416,416)):    #bb[x_cen, y_cen, w, h, x_cen, y_cen, w, h..]
    truth=np.zeros((output_shape))
    box_w= ref[0]/output_shape[0]
    for i in range(0,len(bb),4):
        for j in range(output_shape[0]):
            if bb[i]>= j*box_w  and  bb[i]< (j+1)*box_w:
                a=j
            if bb[i+1]>= j*box_w  and  bb[i+1]< (j+1)*box_w:
                b=j
                
        truth[a,b,0:]= 1,*bb[i:i+4]
        truth[a,b,1]= truth[a,b,1]-(a*box_w)
        truth[a,b,2]= truth[a,b,2]-(b*box_w)
        truth[a,b,1:3]= truth[a,b,1:3]/box_w        
        truth[a,b,3:]= truth[a,b,3:]/ref[1] # [a,b,:]=[1 if plane, 0-1 of where x_cen is in cell, same for y, w/ref width, same for h]
    return truth
    

def prep_pics(img_folder='D:/cnntry/air_data/trainairplane',ref=(416,416)):
    training_data = []
    x_data=np.zeros((200,416, 416, 3))
    y_data=np.zeros((200,13,13,5))
    n=0
    for img in tqdm(os.listdir(img_folder)):
        n=n+1
        if n>200:
            break
        truth= create_truth_arr(conv_bb_coord(rescale_bb(*read_xml(img))))
        path = os.path.join(img_folder,img)
        img = cv2.imread(path)
        img = cv2.resize(img, ref)
        x_data[n-1,...]=np.array(img)
        y_data[n-1,...]=truth
    np.save('x_data.npy', x_data)
    np.save('y_data.npy', y_data)
    return x_data, y_data

def anchors_from_data(num_anchors, img_folder='D:/cnntry/air_data/trainairplane',ref=(416,416), anch_dir='D:\cnntry'):
    if os.path.exists('{}anchors.npy'.format(num_anchors)):
        anchors=np.load('{}anchors.npy'.format(num_anchors))
        bbs_arr=np.load('bounding_boxes.npy')
    elif os.path.exists('bounding_boxes.npy'):
        bbs_arr=np.load('bounding_boxes.npy')
        kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(bbs_arr)
        anchors = kmeans.cluster_centers_
        np.save('{}anchors.npy'.format(num_anchors), anchors)
    else:    
        bbs=[]
        for img in tqdm(os.listdir(img_folder)):
            bb = rescale_bb(*read_xml(img))       
            for i in range(0,len(bb),2):
                bbs.append((bb[i],bb[i+1]))
        bbs_arr = np.asarray(bbs)              #array of bounding boxes
        kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(bbs_arr)
        anchors = kmeans.cluster_centers_    
        np.save('{}anchors.npy'.format(num_anchors), anchors)
        np.save('bounding_boxes.npy', bbs_arr)
        
    return anchors, bbs_arr

#Conversely, if a detector does have a ground-truth box, we want to punish it:

#    when the coordinates are wrong
#    when the confidence score is too low


def custom_loss(y_true, y_pred, no_object_scale=0.5, bb_scale=5.):
    shape = y_true.shape #(13,13,5)
    
    #no object loss 
    no_objects_mask = ma.masked_equal(y_true, 0).mask
    no_object_loss = np.sum((0 - K.sigmoid(y_pred*no_objects_mask)[:,:,0])**2)
    
    # loss from bounding boxes
    bb_loss= np.sum((0 - K.sigmoid(y_pred * ~no_objects_mask)[:,:,1:])**2)

    loss= no_object_scale * no_object_loss + bb_scale * bb_loss
          
    return loss

def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


#anchors, X=(anchors_from_data(4))


#x_data, y_data=prep_pics()

x_data=np.load('xdata.npy')
y_data=np.load('ydata.npy')


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

model.compile('SGD', loss='mean_squared_error')#custom_loss)
#model.summary()




model.fit(x=x_data, y=y_data, batch_size=50, epochs=1, verbose=1, validation_split=0.2, shuffle=True)
model.save('my_model.h5')





*********************************************************** OLD CODE FOR PLAYING WITH COCO INTERFACE





import os
import json
import coco
#import time
#import matplotlib.pyplot as plt
#from matplotlib.collections import PatchCollection
#from matplotlib.patches import Polygon
#import numpy as np
#import copy
#import itertools
#from . import mask as maskUtils
from collections import defaultdict
#import sys

#stuff_train2017.json
annotation_file=os.path.join(os.path.join(os.path.dirname(__file__),'annotations'),'instances_val2017.json')

#{'id': 5, 'name': 'airplane', 'supercategory': 'vehicle'}
dataset = json.load(open(annotation_file, 'r'))

anns,cats,imgs =dict(),dict(),dict()
imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

anns, cats, imgs = {}, {}, {}

if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgToAnns[ann['image_id']].append(ann)
        anns[ann['id']] = ann

if 'images' in dataset:
    for img in dataset['images']:
        imgs[img['id']] = img

if 'categories' in dataset:
    for cat in dataset['categories']:
        cats[cat['id']] = cat

if 'annotations' in dataset and 'categories' in dataset:
    for ann in dataset['annotations']:
        catToImgs[ann['category_id']].append(ann['image_id'])

print('index created!')

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def getImgIds(imgIds=[], catIds=[]):
    '''
    Get img ids that satisfy given filter conditions.
    :param imgIds (int array) : get imgs for given ids
    :param catIds (int array) : get imgs with all given cats
    :return: ids (int array)  : integer array of img ids
    '''
    imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
    catIds = catIds if _isArrayLike(catIds) else [catIds]

    if len(imgIds) == len(catIds) == 0:
        ids = imgs.keys()
    else:
        ids = set(imgIds)
        for i, catId in enumerate(catIds):
            if i == 0 and len(ids) == 0:
                ids = set(catToImgs[catId])
            else:
                ids &= set(catToImgs[catId])
    return list(ids)


for key, value in dataset['info'].items():
            print('{}: {}'.format(key, value))


a=getImgIds(catIds=[5])




************************************************************************* 	EXTRA OLD DATA PREP




import cv2             # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()

wannatrain=True
wannaprep_data=False

TRAIN_DIR = 'D:\cnntry\\train'
TEST_DIR = 'D:\cnntry\\test'

IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes 

def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data    
    
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('D:\cnntry\\test_data.npy', testing_data)
    return testing_data    

if wannaprep_data: train_data = create_train_data()
else: train_data = np.load('D:\cnntry\\train_data.npy') 
 
    
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')



if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train])
X=X.reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)   








