"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import cv2             # working with, mainly resizing, images
from tqdm import tqdm
from lxml import etree
import paramiko
from scipy.ndimage import rotate, interpolation, morphology
from scipy.ndimage.filters import gaussian_filter
import random
import os
import numpy as np
#import numpy.ma as ma
import json, codecs
from keras import layers, models
#from keras.regularizers import l2
from keras import backend as K
from keras.models import model_from_json
import keras
#import tensorflow as tf
'''
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras import models
from keras import layers
from keras.layers import Input
from keras.utils.generic_utils import get_custom_objects
'''

dirname = os.path.dirname(os.path.abspath(__file__))
data_dir= os.path.join(dirname, 'data')

train_pics_dir=os.path.join(data_dir, 'train_pics')
train_annot_dir=os.path.join(data_dir, 'train_annot')

eval_pics_dir=os.path.join(data_dir, 'eval_pics')
eval_annot_dir=os.path.join(data_dir, 'eval_annot')
names='good_pic_names.npy'


def prep_pics(names='', data_dir='', img_folder='', annot_dir='', ref=(416,416), stop=True, stopval=100):
	picnames = np.load(os.path.join(data_dir,names))
	good_names = []
	for i in range(len(picnames)):
		if picnames[i].split(' ')[0]=='ok':
			good_names.append(picnames[i].split(' ')[1]) #just list of names for decent pics
			
	if stop:
		x_data=np.zeros((stopval,416, 416, 3), dtype='uint8')
		y_data=np.zeros((stopval,13,13,5))		
	else:	
		x_data=np.zeros((len(good_names),416, 416, 3), dtype='uint8')
		y_data=np.zeros((len(good_names),13,13,5))
        
	n=0
	for img_name in tqdm(good_names):
		n=n+1
		if stop and n>stopval:
			break
		truth= create_truth_arr(conv_bb_coord(rescale_bb(*read_xml(img_name, annot_dir))))
		img = cv2.imread(os.path.join(img_folder,img_name))
		img = cv2.resize(img, ref)
		x_data[n-1,...]=img
		y_data[n-1,...]=truth
	return x_data, y_data


def prep_pretrain_pics(data_dir='',save_dir='', ref=(416,416), stop=True, startval=0, stopval=100, save=True, load=True):	
	if stop:
		if load:
			try:
				print(os.path.join(save_dir,'pretrain_x{}to{}.npy'.format(startval,stopval)))
				x_data = np.load(os.path.join(save_dir,'pretrain_x{}to{}.npy'.format(startval,stopval)))
				y_data = np.load(os.path.join(save_dir,'pretrain_y{}to{}.npy'.format(startval,stopval)))
				print('\nx_data and y_data loaded\n')
				return x_data, y_data
			except: print('data need to be created')
			
		pic_names = os.listdir(data_dir)
		x_data=np.zeros((stopval-startval,ref[0],ref[1], 3), dtype='uint8')
		shape = np.shape(x_data)
		y_data=np.zeros(shape, dtype='uint8')	
		
		print('2')    
		end=stopval-startval
		n=0
		for pic_name in tqdm(pic_names):
			n+=1
			if stop and n>end: break			
			img = cv2.imread(os.path.join(data_dir, pic_name))
			img = cv2.resize(img, ref)		
			x_data[n-1,...]=img
			y_data[n-1,...]=img
		
		print('3')
		x_hf = np.random.uniform(low=0, high=255, size=shape).astype('uint8')
		y_hf = np.zeros(shape, dtype='uint8')
		
		size = 13
		x_lf = np.random.uniform(low=0, high=255, size=(shape[0],size,size,3)).astype('uint8')
		x_lf = interpolation.zoom(x_lf, [1, ref[0]/size ,ref[1]/size , 1], order=0, mode='constant', cval=0)
		y_lf = np.zeros(shape, dtype='uint8')		
		for i in range(3) :x_lf[:,:,i] = gaussian_filter(x_lf[:,:,i] , sigma=16)
		print('4')
		x_data = np.concatenate((x_data, x_hf, x_lf),axis=0)
		y_data = np.concatenate((y_data, y_hf, y_lf),axis=0)
		
		rng_state = np.random.get_state()
		np.random.shuffle(x_data)
		np.random.set_state(rng_state)
		np.random.shuffle(y_data)
		print('5')
		if save:
			np.save(os.path.join(save_dir,'pretrain_x{}to{}.npy'.format(startval,stopval)), x_data)  
			np.save(os.path.join(save_dir,'pretrain_y{}to{}.npy'.format(startval,stopval)), y_data)
			print('\nx_data and y_data saved\n')
		
		return x_data, y_data

	else:	
		pic_names = os.listdir(data_dir)
		chunks_size=1000 
		#chunks = 3
		parts = int(len(pic_names)  / chunks_size)
		if not len(pic_names)%chunks_size==0 : parts+=1
		n=-1
		for part in range(parts):
			x_data=np.zeros((chunks_size,416, 416, 3), dtype='uint8')
			y_data=np.zeros((chunks_size,416, 416, 3), dtype='uint8')

			print(n, part)
			for i in tqdm(range(chunks_size)):					
				n+=1
				if n>(len(pic_names)-1): 
					save=False
					break
				img = cv2.imread(os.path.join(data_dir, pic_names[n]))
				img = cv2.resize(img, ref)	
				try:
					x_data[n-chunks_size*(part),...]=img
					y_data[n-chunks_size*(part),...]=img
				except Exception as e:
					print('Error: ',e)
		
			print(n,chunks_size*(part+1),len(pic_names) )
			x_hf = np.random.uniform(low=0, high=255, size=(chunks_size,ref[0],ref[1],3)).astype('uint8')
			y_hf = np.zeros((chunks_size, 416, 416, 3), dtype='uint8')
			
			size = 4
			x_lf = np.random.uniform(low=0, high=255, size=(chunks_size,size,size,3)).astype('uint8')
			x_lf = interpolation.zoom(x_lf, [1, ref[0]/size, ref[1]/size, 1], order=0, mode='constant', cval=0)
			y_lf = np.zeros((chunks_size, 416, 416, 3), dtype='uint8')		
			for i in range(3) :x_lf[:,:,i] = gaussian_filter(x_lf[:,:,i] , sigma=30)

			x_data = np.concatenate((x_data, x_hf, x_lf),axis=0)
			y_data = np.concatenate((y_data, y_hf, y_lf),axis=0)
			
			rng_state = np.random.get_state()
			np.random.shuffle(x_data)
			np.random.set_state(rng_state)
			np.random.shuffle(y_data)
			print('shuffled')
			if save:
				np.save(os.path.join(save_dir,'pretrain_x{}part.npy'.format(part)), x_data)  
				np.save(os.path.join(save_dir,'pretrain_y{}part.npy'.format(part)), y_data)
				print('part{} saved\n'.format(part))					




def read_xml(img_name, annot_dir):
    annot_label = img_name.split('.')[0]+'.xml'           #xml file name
    annot_xml = os.path.join(annot_dir,annot_label)  #xml file full path
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

def custom_loss(y_true, y_pred):#, no_object_scale=0.5, bb_scale=5., object_scale=5.): #(0.5,5,5)
	no_objects_mask = K.cast(K.equal(y_true, 0),K.floatx())
	objects_mask = K.cast(K.not_equal(y_true, 0),K.floatx())

	#no object loss 	
	no_object_loss = K.sum(K.square(0 - (y_pred*no_objects_mask)[:,:,:,0]))
	    
	#object loss 
	object_loss = K.sum(K.square(((1 - y_pred) * objects_mask)[:,:,:,0]))

	# loss from bounding boxes
	bb_loss= K.sum(K.square(y_true[:,:,:,1:] - (y_pred * objects_mask)[:,:,:,1:]))
	
	loss= 1 * no_object_loss  +  5 * bb_loss  +  5 * object_loss      
	return loss



def anchors_from_data(num_anchors, img_folder, annot_dir, ref=(416,416)):
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
            bb = rescale_bb(*read_xml(img, annot_dir))       
            for i in range(0,len(bb),2):
                bbs.append((bb[i],bb[i+1]))
        bbs_arr = np.asarray(bbs)              #array of bounding boxes
        kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(bbs_arr)
        anchors = kmeans.cluster_centers_    
        np.save('{}anchors.npy'.format(num_anchors), anchors)
        np.save('bounding_boxes.npy', bbs_arr)        
    return anchors, bbs_arr




def prep_pic(img, ref=(416,416), return_OGimg=True, pic_dir='eh'):  #img_folder='D:/cnntry/air_data/trainairplane'
    x_data=np.zeros((1,ref[0], ref[1], 3))
    #y_data=np.zeros((13,13,5))
    #truth= create_truth_arr(conv_bb_coord(rescale_bb(*read_xml(img))))
    #path = os.path.join(img_folder,'\\')
    path = os.path.join(pic_dir,img)
    OGimg = cv2.imread(path)
    img = cv2.resize(OGimg, ref)
    x_data[0] = np.array(img)
    if return_OGimg:
        return x_data, OGimg  #np.array of scaled img,  OG cv2 image
    else: return x_data


def create_truth_arr(bb, output_shape=(13,13,5), ref=(416,416)):    #bb[x_cen, y_cen, w, h, x_cen, y_cen, w, h..]
    truth=np.zeros((output_shape))
    box_w= ref[0]/output_shape[0]
    for i in range(0,len(bb),4):
        for j in range(output_shape[0]):
            if bb[i]>= j*box_w  and  bb[i]< (j+1)*box_w:
                a=j
            if bb[i+1]>= j*box_w  and  bb[i+1]< (j+1)*box_w:
                b=j
                
        truth[a,b,0:]= 1, bb[i], bb[i+1], bb[i+2], bb[i+3]
        truth[a,b,1]= truth[a,b,1]-(a*box_w)
        truth[a,b,2]= truth[a,b,2]-(b*box_w)
        truth[a,b,1:3]= truth[a,b,1:3]/box_w        
        truth[a,b,3:]= truth[a,b,3:]/ref[1] # [a,b,:]=[1 if plane, 0-1 of where x_cen is in cell, same for y, w/ref width, same for h]
    return truth

def ypred_to_bbs(y_pred, OGshape, threshold, ref=(416,416)):
    shape=y_pred.shape
    scale_w = OGshape[0]/ref[0]
    scale_h = OGshape[1]/ref[1]
    detector_w = ref[0]/shape[0] 
    detector_h = ref[1]/shape[1]
    bbs=[]  #[#detec_x, #detec_y,  0-1 of where x_cen is in cell,   same for y,   w/ref width,   h/ref height]]
    for x in range(shape[0]):
        for y in range(shape[1]):
            if y_pred[x,y,0]>=threshold:
                bbs.append([x,y,*y_pred[x,y,1:]])
    b_b=[]
    for i in bbs:
        x_cen = i[2]*detector_w + i[0]*detector_w
        y_cen = i[3]*detector_h + i[1]*detector_h
        w = i[4]*ref[0]
        h = i[5]*ref[1]
        #print(x_cen, y_cen,w,h)        
        b_b.append([(x_cen-w/2)*scale_w,  (y_cen-h/2)*scale_h,  (x_cen+w/2)*scale_w,  (y_cen+h/2)*scale_h])
        #print(b_b[-1])
        for i in range(len(b_b[-1])):
            if b_b[-1][i]<0.: b_b[-1][i]=0 
            else: b_b[-1][i] = int(b_b[-1][i])
        #print(b_b)
    return b_b  # [x-topleft, y-topleft, x-bottomright, y-bottomright ..repeat]

def viz_output(model, img, t=3, threshold=0.3, plot=False, pic_dir='', annot_dir=''):
    x_data, OGimg = prep_pic(img, pic_dir=pic_dir, return_OGimg=True)
    y_pred = model.predict(x_data)
    #print(y_pred[0].shape)
    img_w, img_h, og_bbs = read_xml(img, annot_dir=annot_dir)  #[x-topleft, y-topleft, x-bottomright, y-bottomright ..repeat]
    pred_bbs = ypred_to_bbs(y_pred[0], OGimg.shape, threshold) # [x-topleft, y-topleft, x-bottomright, y-bottomright ..repeat]
    if plot:
        OGimg = cv2.cvtColor(OGimg, cv2.COLOR_BGR2RGB)
        for i in range(0, len(og_bbs), 4):
            tl=(og_bbs[i+0],og_bbs[i+1])
            br=(og_bbs[i+2],og_bbs[i+3])
            OGimg = cv2.rectangle(OGimg,tl,br,(255,0,0),t) #RGB OG bounding boxes are red
        for bb in pred_bbs:
            tl=(bb[0],bb[1])
            br=(bb[2],bb[3])
            OGimg = cv2.rectangle(OGimg,tl,br,(0,255,0),t) #RGB predicted bounding boxes are green    
        plt.imshow(OGimg)   
        plt.show
    else: return x_data, y_pred, og_bbs, pred_bbs, # og_bbs is a straight list and pred_bbs is a list of lists

def eval_centers(model, threshold=0.99, pic_dir='', annot_dir='' ):
    true_cen=[] #[[xtrue,ytrue]]
    pred_cen=[] #[[x,y]]
    for img in tqdm(os.listdir(pic_dir)):
        x_data, y_pred, og_bbs, pred_bbs = viz_output(model, img, threshold=threshold, pic_dir=pic_dir, annot_dir=annot_dir )
        for i in range(0, len(og_bbs), 4):
            true_cen.append([(og_bbs[i+0]+og_bbs[i+2])/2 , (og_bbs[i+1]+og_bbs[i+3])/2])
        for bb in pred_bbs:
            pred_cen.append([(bb[0]+bb[2])/2 , (bb[1]+bb[3])/2])    
    true_cen = np.asarray(true_cen)
    pred_cen = np.asarray(pred_cen)
    plt.scatter(true_cen[:,0],true_cen[:,1], c='r')
    plt.scatter(pred_cen[:,0],pred_cen[:,1], c='g')
    plt.title('center distribution')
    plt.show
    
    return true_cen

def load_model(name):
	json_file = open("{}.json".format(name), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("{}weights.h5".format(name))
	return model

def save_model(model, name):
	#saves only architecture and weights
	# serialize model to JSON
	model_json = model.to_json()
	with open("{}.json".format(name), "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("{}weights.h5".format(name),overwrite=True)	
	
def load_model_w_optim(name):
	model = keras.models.load_model('{}.h5'.format(name))
	return model

def save_model_w_optim(model, name):
	#saves architecture, weights, what you passed to compile and optimizer
	model.save('{}.h5'.format(name),overwrite=True)

def saveHist(path, history):
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4) 

def loadHist(path):
	n = {} # set history to empty
	if os.path.exists(path): # reload history if it exists
		with codecs.open(path, 'r', encoding='utf-8') as f:
			n = json.loads(f.read())

	return n		
	
def time_print(start, stop, mtext):
	now_time = time.process_time()
	if start==None and stop==None:
		return now_time
	elif stop==None:
		t = int((now_time - start) * 100) / 100.
		print('T+'+str(int(t/3600))+':'+str(int(t%3600/60))+':'+str(t%60),' process took:',\
		      str(int(t/3600))+':'+str(int(t%3600/60))+':'+str(t%60), 'sec -- {}'.format(mtext))
		return now_time
	else:
		t = int((now_time - start) * 100) / 100.
		pt = int((now_time - stop) * 100) / 100.
		print('T+'+str(int(t/3600))+':'+str(int(t%3600/60))+':'+str(t%60),' process took:',\
		      str(int(pt/3600))+':'+str(int(pt%3600/60))+':'+str(pt%60), 'sec -- {}'.format(mtext))
		return now_time

def setup_ssh():
	ssh_client=paramiko.SSHClient()
	ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	hname='131.180.117.41'
	uname='andrej'
	ssh_client.connect(hname,22,username=uname, password='gamegame')
	return ssh_client

def download_trained_script(model_name, where_to='D://cnn//', where_from='/home/andrej/'):
	ssh_client = setup_ssh()	
	ftp_client=ssh_client.open_sftp()
	ftp_client.get('{}{}weights.h5'.format(where_from, model_name),'{}d{}weights.h5'.format(where_to, model_name))
	ftp_client.get('{}{}.json'.format(where_from, model_name),'{}d{}.json'.format(where_to, model_name))
	ftp_client.close()
	
def download_files_script(*file_names, where_to='D://cnn//', where_from='/home/andrej/'):
	ssh_client = setup_ssh()	
	ftp_client=ssh_client.open_sftp()
	for file_name in file_names:
		ftp_client.get('{}{}'.format(where_from, file_name),'{}{}'.format(where_to, file_name))
	ftp_client.close()

def upload_py_files_script(*args, where_from='D://cnn//', where_to='/home/andrej/'):
	ssh_client = setup_ssh()	
	ftp_client=ssh_client.open_sftp()
	for file_name in args:	
		try:
			ftp_client.remove('{}{}'.format(where_to, file_name))
		except:
			print('no file removed')
		ftp_client.put('{}{}'.format(where_from, file_name),'{}{}'.format(where_to, file_name))
	ftp_client.close()



class my_models():
	'Generates and compiles my models'
	def __init__(self, optimizer='Nadam', loss='mean_squared_error' ):
		'Initialization'
		self.optim = optimizer
		self.loss = loss

	
	def yolo_arch(self, summary=True):
		model = models.Sequential()
		model.add(layers.Conv2D(16, 3, strides=(1, 1), padding='same',input_shape=(416, 416, 3)))
		model.add(layers.LeakyReLU(alpha=0.1))
		
		filters=[32,64,128,256,512]#,1024]
		for fil in filters:
		    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
		    model.add(layers.Conv2D(fil, 3, strides=(1, 1), padding='same')) 
		    model.add(layers.LeakyReLU(alpha=0.1))
		
		model.add(layers.Conv2D(512, 3, strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Conv2D(5, 3, strides=(1, 1), padding='same'))  
		
		model.compile('Nadam', loss=custom_loss, sample_weight_mode = None)
		model.compile(self.optim, loss=self.loss, sample_weight_mode = None)#  'mean_squared_error'
		if summary: model.summary()
		return model
	




class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, bcg_paths = None, plane_paths = None, steps_per_epoch=10, batch_size=16, \
			     ref_in=(416,416,3), ref_out=(13,13,5), testing = False,):        
		'Initialization'
		self.steps_per_epoch = steps_per_epoch
		self.batch_size = batch_size
		self.bcg_paths = bcg_paths
		self.plane_paths = plane_paths
		self.ref_in = ref_in
		self.ref_out = ref_out
		self.on_epoch_end()
		self.testing = testing
		
		self.box_w = self.ref_in[0]/self.ref_out[0]
		self.box_h = self.ref_in[1]/self.ref_out[1]
		

	def __len__(self):
		'Denotes the number of batches per epoch'
		return self.steps_per_epoch

	def __getitem__(self, index):
		'Generate one batch of data'
		x_data = np.empty((self.batch_size,*self.ref_in ))
		y_data = np.empty((self.batch_size,*self.ref_out))
		ch_bcg_names = random.sample(self.bcg_paths, self.batch_size)
		
		for i in range(len(ch_bcg_names)):
			x_data[i], y_data[i]= self.gen_training_pic(ch_bcg_names[i])
		
		return (x_data/255, y_data)
	
	def gen_training_pic(self, bcg_path):
		bcg = cv2.imread(bcg_path)
		bcg = cv2.cvtColor(bcg, cv2.COLOR_BGR2RGB)	
		r_n_planes= np.random.choice(a=[1, 2, 3, 4, 5], p=[150/200, 37/200, 9/200, 3/200, 1/200]) #add 0
		for i in range(r_n_planes):
			plane_path = random.choice(self.plane_paths)
			plane = np.load(plane_path)	
			# Get random values
			r_unif = np.random.uniform(low=0.0, high=1.0, size=(3,1)) 
			
			angs = np.linspace(-90, 90, 181).astype('int32')
			plane_angs_prob = np.ones(angs.shape)
			plane_angs_prob[[0,-1]] = 10
			plane_angs_prob[90] = 40
			plane_angs_prob = plane_angs_prob/np.sum(plane_angs_prob)
			plane_ang = np.random.choice(a=angs, p=plane_angs_prob)
			
			bcg_angs_prob = np.ones(angs.shape)
			bcg_angs_prob[[0,-1]] = 10
			bcg_angs_prob[90] = 30
			bcg_angs_prob[82:99] = 10
			bcg_angs_prob = bcg_angs_prob/np.sum(bcg_angs_prob)			
			bcg_ang = np.random.choice(a=angs, p=bcg_angs_prob)
			
			args = [plane, bcg, r_unif[0], r_unif[1]]	
			merge_kwargs={'fliph_plane': np.random.choice(a=[False, True], p=[0.5, 0.5]),
					      'plane_angle': plane_ang,
						  'scale': 		 np.random.uniform(low=0.05, high=0.7, size=(1,1)).item(),
						  'fliph_bcg':   np.random.choice(a=[False, True], p=[0.5, 0.5]),
						  'bcg_angle': 	 bcg_ang
						  }

			if  i==0: x_d, y_t = self.merge(*args, **merge_kwargs)
			elif i>0: 
				args[1] = x_d		
				x_d, y_t = self.merge(*args, **merge_kwargs, y_true=y_t)


		x_d = cv2.resize(x_d, self.ref_in[0:2])
		return x_d, y_t
		

	def merge(self, plane, bcg, x_pos_r, y_pos_r,\
			  fliph_plane=False, plane_angle=0, scale=1, \
			  fliph_bcg = False, bcg_angle=0, y_true=None, \
			  min_pix_plane=10):
		#1.line  = generic arguments
		#2.line  = aircraft specific inputs
		#3.line  = background specific inputs
		#4.line  = most likely not necesarry to specify kwargs
		
		if y_true is None:
			y_true=np.zeros((self.ref_out))
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
		plane_mask = np.dstack((plane_mask, plane_mask, plane_mask))
		plane = plane[rmin:rmax+1,cmin:cmax+1,:]
		p_sha = plane.shape
		bcg_sha=bcg.shape
		
		#scaling the plane with its mask
		fin_r = scale * min([bcg_sha[0]/p_sha[0], bcg_sha[1]/p_sha[1]]) # scaling with 1 being the background width	
		if fin_r*p_sha[0] * (self.ref_in[0]/bcg_sha[0])<min_pix_plane:
			fin_r = fin_r*min_pix_plane/(fin_r*p_sha[0] * (self.ref_in[0]/bcg_sha[0]))		
		if fin_r*p_sha[1] * (self.ref_in[1]/bcg_sha[1])<min_pix_plane:
			fin_r = fin_r*min_pix_plane/(fin_r*p_sha[1] * (self.ref_in[1]/bcg_sha[1]))
		plane = interpolation.zoom(plane, [fin_r ,fin_r , 1], order=4, mode='constant', cval=0)
		plane_mask = interpolation.zoom(plane_mask, [fin_r, fin_r , 1], order=0, mode='constant', cval=False)
	

		# Setting up values for truth array
		p_sha=plane.shape
		x_pix = int( x_pos_r * (bcg_sha[0] - p_sha[0]) ) # x and y coordinate of top left plane corner in background coordinates
		y_pix = int( y_pos_r * (bcg_sha[1] - p_sha[1]) )
		w_ratio= self.ref_in[0] / bcg_sha[0]
		h_ratio= self.ref_in[1] / bcg_sha[1]
		x_plane_cen = int( (x_pix+p_sha[0]/2)*w_ratio ) # x and y coordinate of center point for plane in ref coordinates
		y_plane_cen = int( (y_pix+p_sha[1]/2)*h_ratio )
		
		
		for j in range(self.ref_out[0]):
			if x_plane_cen>= j*self.box_w  and  x_plane_cen< (j+1)*self.box_w:
				a=j
			if y_plane_cen>= j*self.box_h  and  y_plane_cen< (j+1)*self.box_h:
				b=j
				
		if (not no_ac_in) and (y_true[a,b,0]==1): # the detector would be responsible for 2 aircraft, so just ignore the new ac
			return bcg, y_true
		else:
			# Filling truth array            
			y_true[a,b,0:]= 1, x_plane_cen, y_plane_cen, p_sha[0]/bcg_sha[0], p_sha[1]/bcg_sha[1]
			y_true[a,b,1]= (y_true[a,b,1]-(a*self.box_w))/self.box_w 
			y_true[a,b,2]= (y_true[a,b,2]-(b*self.box_w))/self.box_w     
		
			#pasting in aircraft
			bcg[x_pix:x_pix+p_sha[0], y_pix:y_pix+p_sha[1]] = bcg[x_pix:x_pix+p_sha[0], y_pix:y_pix+p_sha[1]] * ~plane_mask  +  plane * plane_mask
			x_data=bcg
			return x_data, y_true





	
if __name__ == '__main__':
	dirname = os.path.dirname(os.path.abspath(__file__))
	data_dir= os.path.join(dirname, 'data')
	bcg_dir = os.path.join(data_dir, 'all_backgrounds')	
	plane_dir = os.path.join(data_dir, 'npy_cut_planes_train')	


