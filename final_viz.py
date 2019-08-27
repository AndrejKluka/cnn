"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""
import time
import functions as fn
import filter_viz
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#from keras.applications import vgg16
import keras
import tensorflow as tf
import keras.backend as K



class Visualization():
	''' 
	Class with many varied and often disconected fuctions related to model evaluation/ visualization,
	functions starting with '_' are internal class functions, the rest is meant for use
	'''
	def __init__(self, model= None, optim = 'Adam', loss_function = fn.custom_loss,
			     ref_x=(416,416,3), ref_y=(13,13,5), 
				 threshold= 1e-6, plot_window=[250, 100, 1280, 824]):
		self.ref_x = ref_x
		self.ref_y = ref_y
		self.threshold = threshold
		self.plot_window = plot_window
		self.OGshape = ref_x
		self.model = model
		self.optim = optim
		self.loss_function = loss_function


	def load_model(self, name = 'best_prev2.h5', from_server = True, to_return = False, 
				   where_from= '/home/andrej/', where_to= 'D://cnn//'):
		if name.split('_')[0] == 'best':
			names = [name]
		else:
			names = [name.split('.')[0]+'.json', name.split('.')[0]+'weights.h5']
			
		if from_server:
			path = where_to
			ssh_client = fn.setup_ssh()	
			ftp_client = ssh_client.open_sftp()			
			for file_name in names:
				ftp_client.get('{}{}'.format(where_from, file_name),'{}{}'.format(where_to, file_name))	
			ftp_client.close()
		else :
			path = where_from
			

		if len(names) == 1:
			#print(path+names[0])
			keras.losses.custom_loss = fn.custom_loss
			self.model = keras.models.load_model(path+names[0])
		else:
			json_file = open(path+names[0], 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			self.model = keras.models.model_from_json(loaded_model_json)
			self.model.load_weights(path+names[1])
		self.model.compile(self.optim, loss=self.loss_function)
		if to_return:
			return self.model
				
	def _y_to_bbs(self, y, ret_scores = False):
		shape = y.shape
		scale_w = self.OGshape[0]/self.ref_x[0]
		scale_h = self.OGshape[1]/self.ref_x[1]
		detector_w = self.ref_x[0]/shape[0] 
		detector_h = self.ref_x[1]/shape[1]		
		bbs=[]  #[#detec_x, #detec_y,  0-1 of where x_cen is in cell,   same for y,   w/ref width,   h/ref height]]
		scores = []
		for x_index in range(shape[0]):
			for y_index in range(shape[1]):
				if y[x_index, y_index, 0]>=self.threshold:
					bbs.append([x_index, y_index, *y[x_index, y_index, 1:]])
					scores.append(y[x_index, y_index, 0])
		b_b=[]
		for i in bbs:
			x_cen = i[2]*detector_w + i[0]*detector_w
			y_cen = i[3]*detector_h + i[1]*detector_h
			w = i[4]*self.ref_x[0]
			h = i[5]*self.ref_x[1]
			#print(x_cen, y_cen,w,h)        
			b_b.append([(x_cen-w/2)*scale_w,  (y_cen-h/2)*scale_h,  (x_cen+w/2)*scale_w,  (y_cen+h/2)*scale_h])
			#print(b_b[-1])
			for i in range(len(b_b[-1])):
				if b_b[-1][i]< 0. : 
					b_b[-1][i]=0 
				elif b_b[-1][i]> self.ref_x[0]: 
					b_b[-1][i]= self.ref_x[0]-1
				else: b_b[-1][i] = int(b_b[-1][i])
			#print(b_b)
		if ret_scores:
			return b_b, scores
		else: return b_b  # [x-topleft, y-topleft, x-bottomright, y-bottomright ..repeat]

	def _bbs_to_mask(self, bbs):
		array = np.zeros(self.OGshape[0:2])
		#print(bbs)
		try:
			for bb in bbs:
				xtl, ytl, xbr, ybr = bb[0], bb[1], bb[2], bb[3]
				array[xtl, ytl:ybr] = 1
				array[xbr, ytl:ybr] = 1
				array[xtl:xbr, ytl] = 1
				array[xtl:xbr, ybr] = 1
			mask = np.ma.masked_not_equal(array, 0).mask
			return mask.reshape((self.OGshape[0], self.OGshape[1], 1))
		except:
			np.zeros((self.OGshape[0], self.OGshape[1], 1))
		
	def _put_mask_on_pic(self, pic, mask, colour):
		try:
			pic = pic * ~mask
			for i in range(3):
				pic[:,:,i] = pic[:,:,i] + mask.reshape(self.OGshape[0:2])*colour[i]
			return pic.astype('uint8')
		except:
			return pic
	
	def plot_x_y(self, x, y, threshold = None, colour = (0,255,0)):
		if threshold is not None: self.threshold = threshold
		x.astype('uint8')
		self.OGshape = x.shape
		y_mask = self._bbs_to_mask( self._y_to_bbs(y))
		pic = self._put_mask_on_pic(x, y_mask, colour)
		self._plot_array(pic) 

	def get_x_y_y_pic(self, x, y_true, y_pred, thr = None, colour =[(255,0,0), (0,255,0)]):
		if thr is not None: self.threshold = thr
		x.astype('uint8')
		self.OGshape = x.shape
		
		y_true_mask = self._bbs_to_mask( self._y_to_bbs(y_true))
		pic = self._put_mask_on_pic(x, y_true_mask, colour[0])
		
		bbs, scores = self._y_to_bbs(y_pred, ret_scores = True)
		y_pred_mask = self._bbs_to_mask(bbs)
		pic = self._put_mask_on_pic(pic, y_pred_mask, colour[1])	
		for i in range(len(bbs)):
			try:
				bb = (bbs[i][1], bbs[i][0]-5)
				pic = cv2.putText(pic, '{:.1f}%'.format(scores[i]*100), bb, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour[1], 1)
			except Exception as e: print('Error',e)
		return pic

	def plot_n_pics(self, xx_data, yy_data, yy_pred, n=(0,1), w=1, h=1, save_name=None, thr = None, colour =[(255,0,0), (0,255,0)]):
		a = np.zeros(( h*self.ref_x[1], w*self.ref_x[0], self.ref_x[2]), dtype='uint8')
		i=n[0]
		for wi in range(w):
			for hi in range(h):
				if i < n[1]:
					pic = self.get_x_y_y_pic(xx_data[i], yy_data[i], yy_pred[i], thr = thr, colour = colour)
					a[ hi*self.ref_x[1] : (hi+1)*self.ref_x[1], wi*self.ref_x[0] : (wi+1)*self.ref_x[0], :] = pic
					i += 1
		self._plot_array(a)
		if save_name is not None:
			a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
			cv2.imwrite(save_name, a)		
		return a
	
	def plot_file(self, path):
		try:
			if path[-3:]=='npy':
				pic = np.load(path)
			else: 
				pic = cv2.imread(path)
				pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)	
		except Exception as e: print('Error: ',e)
		self._plot_array(pic)

	def model_summary(self):
		if self.model is not None: self.model.summary()
		else: print('No model loaded') 


	def viz_filter(self, LAYER_NAME='conv2d_5', upscaling_steps=12, filter_range=(0, 16)):
		try:
			filter_viz.visualize_layer(self.model, LAYER_NAME, upscaling_steps=upscaling_steps, filter_range=filter_range)
		except Exception as e: print('Error: ',e)
		
		
	def _plot_array(self, array):	
		if array.dtype=='bool':
			array = array.astype('float')
		plt.figure()
		mngr = plt.get_current_fig_manager()
		mngr.window.setGeometry(*self.plot_window)
		plt.imshow(array)

	def get_them_errors(self, y_true, y_pred):
		import mAPcalc as mAP
		gt_boxes = {}
		pred_boxes = {}
		
		for i in range(y_true.shape[0]):
			name = 'img{}'.format(i)
			# [[x-topleft, y-topleft, x-bottomright, y-bottomright],[ ..repeat]]
			true_boxes_list, true_scores = self._y_to_bbs(y_true[i], ret_scores = True)
			gt_boxes[name] = true_boxes_list
			
			pred_boxes_list, pred_scores = self._y_to_bbs(y_pred[i], ret_scores = True)
			pred_boxes[name] = {'boxes':pred_boxes_list, 'scores':pred_scores}

		iou_thr = 0.5
		start_time = time.time()
		data = mAP.get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
		end_time = time.time()
		print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
		print('avg precision: {:.4f}'.format(data['avg_prec']))
	
		start_time = time.time()
		ax = None
		avg_precs = []
		iou_thrs = []
		for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
			data = mAP.get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
			avg_precs.append(data['avg_prec'])
			iou_thrs.append(iou_thr)
	
			precisions = data['precisions']
			recalls = data['recalls']
			ax = mAP.plot_pr_curve(
				precisions, recalls, label='{:.2f}'.format(iou_thr), color= mAP.COLORS[idx*2], ax=ax)
	
		# prettify for printing:
		avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
		iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
		print('map: {:.2f}'.format(100*np.mean(avg_precs)))
		print('avg precs: ', avg_precs)
		print('iou_thrs:  ', iou_thrs)
		plt.legend(loc='upper right', title='IOU Thr', frameon=True)
		for xval in np.linspace(0.0, 1.0, 11):
			plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
		end_time = time.time()
		print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
		plt.show()

	
	def get_flops(model):
	    run_meta = tf.RunMetadata()
	    opts = tf.profiler.ProfileOptionBuilder.float_operation()
	
	    # We use the Keras session graph in the call to the profiler.
	    flops = tf.profiler.profile(graph=K.get_session().graph,
	                                run_meta=run_meta, cmd='op', options=opts)
	
	    return flops.total_float_ops  # Prints the "flops" of the model.


if __name__ == '__main__':	
		
	Viz = Visualization(model= None,
						optim = 'Adam',
						loss_function = fn.custom_loss,
						ref_x=(416,416,3),
						ref_y=(13,13,5),
						threshold= 1e-4,
						plot_window=[250, 100, 1280, 824])
	
	model = Viz.load_model(name = 'best_workingL48.h5', 
						   from_server = False,
						   to_return = True,
						   where_from= 'D://cnn//',
						   where_to= '')
	
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
	
	
	#set up paths
	optim='Adam'
	load_name = 'best_prev2'#'best_pretrain_model'#'pretrain_model'
	dirname = os.path.dirname(os.path.abspath(__file__))
	data_dir= os.path.join(dirname, 'data')
	
	
	
	#x_data = np.load('pretrain_x0to100.npy')[:20]
	#y_data = np.load('pretrain_y0to100.npy')[:20]
	#y_pred = model.predict(x_data)
	
	''' GENERATING DATA
	xny = eval_generator[0]
	xx_data = xny[0]
	yy_data = xny[1]
	yy_pred = model.predict(xx_data)'''
	
	# LOADING DATA
	xx_data = np.load('gen_val_x.npy')
	yy_data = np.load('gen_val_y.npy')
	yy_pred = model.predict(xx_data[:].astype('float32')/255)
	
	
	'''  Maximum activation of a filter visualization '''
	#Viz.model_summary()
	#Viz.viz_filter(LAYER_NAME='conv2d_5', upscaling_steps=12, filter_range=(25, 50))
	
	''' Saving images with predicted and GT boxes '''
	#a=Viz.plot_n_pics(xx_data, yy_data, yy_pred, n=(0,6), w=3, h=2, save_name='predicts.png', thr = 0.6, colour =[(255,0,0), (0,255,0)])		
	
	''' Showing images with predicted and GT boxes '''
	for i in range(5):
		pic = Viz.get_x_y_y_pic(xx_data[i], yy_data[i], yy_pred[i], thr = 0.6)
		Viz._plot_array(pic)
		
	''' Convinience functions for showing only GT or pred bbs '''
	def io(i):
		Viz.plot_x_y(xx_data[i].astype('uint8'), yy_data[i], threshold = 0.01)
	def iu(i,t=0.01):
		Viz.plot_x_y(xx_data[i].astype('uint8'), yy_pred[i], threshold = t)
	#	for i in range(5): iu(i, t=0.5)
	#	for i in range(5): io(i)	
		
	
	
	''' Getting mAP '''
	#Viz.get_them_errors(yy_data, yy_pred)		
	'''
	Model Loss 58
	Single IoU calculation took 19.4720 secs
	avg precision: 0.5171
	map: 20.10
	avg precs:  [0.5171, 0.4386, 0.3568, 0.2748, 0.1934, 0.1407, 0.0591, 0.0248, 0.004, 0.0006]
	iou_thrs:   [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
	Plotting and calculating mAP takes 184.2679 secs
	
	Model Loss 48.5
	Single IoU calculation took 18.8810 secs
	avg precision: 0.5579
	map: 24.32
	avg precs:  [0.5579, 0.5033, 0.4306, 0.346, 0.25, 0.1828, 0.1041, 0.0443, 0.013, 0.0005]
	iou_thrs:   [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
	Plotting and calculating mAP takes 192.6751 secs
	'''
	
	
		
	''' Timing the inference time '''
	#start = fn.time_print(None, None, None)
	#yy_pred = model.predict(xx_data)
	#stop = fn.time_print(start, None, 'Predictions predicted')
	#print(xx_data.shape)
	#print('{:.4f} secs per img)\n'.format(stop - start))
	
	
	''' Number of FLOPs in a model '''
	#print(Viz.get_flops(model))
	
	
	
	
	
	''' VIZ filter numbers
	Plots filters from specified layer into a heatmap
	'''
	if False:
		# np.shape(x.get_weights()[0]) 	 	#[filterx, filtery, input 3rd dim, n of filters]
		#model = vgg16.VGG16(weights='imagenet', include_top=False) # VGG for testing purposes
		x = model.layers[0]
		filters = 16
		subplot_rows=1
		subplot_columns=16
		
		input_z = np.shape(x.get_weights()[0])[2]
		filt_dims = np.shape(x.get_weights()[0])
		
		# subplot(2,3,4) 2 rows,  3 columns,  4th element(2nd row 1st column)
		x1w = x.get_weights()[0][:,:,:,:]
		n=1
		#looping over all specifiec filters
		for n_filt in range(1,filters+1):
			plt.subplot(subplot_rows, subplot_columns, n)
			n+=1
			z_flattened = x1w[:,:,0,n_filt-1]
			for z_layer in range(input_z-1):
				z_flattened =  np.concatenate((z_flattened, x1w[:, :, z_layer+1, n_filt-1]), axis=0)
			plt.imshow(z_flattened, interpolation="nearest", cmap="coolwarm") #nearest, bicubic
		plt.show()











