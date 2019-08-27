"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""
import time, os, warnings, keras
import functions as fn
import numpy as np
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
import adjusted_callbacks as fwc
from keras import callbacks
import keras.backend as K


		
		
if __name__ == '__main__':	
	# Datasets
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
	hist = {'loss': [], 'val_loss': []}
	
	# Generators
	train_params = {'bcg_paths': bcg_names[int(len(bcg_names)*0.05):],
				    'plane_paths': plane_names[int(len(bcg_names)*0.05):],
					'steps_per_epoch': 5,
					'batch_size': 16,
					'ref_in': (416,416,3),
					'ref_out': (13,13,5)}
	training_generator = fn.DataGenerator( **train_params)
	
	# Validation dataset
	valx = np.load('gen_val_x.npy').astype('float32')/255   
	valy = np.load('gen_val_y.npy').astype('float32')

	# Setup model
	keras.backend.clear_session()
	keras.losses.custom_loss = fn.custom_loss
	optim = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=True)
	Model = fn.my_models(optimizer=optim, loss=fn.custom_loss)
	
	cont = True
	if not cont:
		#model = fn.load_model('best_initiate')
		#model.compile(optim, loss=fn.custom_loss)
		model = Model.my_mess5(summary=True)
		#model = yolo_arch_v3(summary= True)	
		#model = fn.load_model_w_optim('best_first_workingL56')
		
	if cont:
		model = fn.load_model_w_optim('best_gen_model')
		hist = fn.loadHist('./best_gen_hist.json')		
		#print(hist)

	# Respecify Learning Rate
	print(K.get_value(model.optimizer.lr))
	K.set_value(model.optimizer.lr, 0.001)
	print(K.get_value(model.optimizer.lr),'\n','\n','\n','\n')


	
	# Setup callbacks
	if hist['val_loss']:
		last_best = min(hist['val_loss'])
		wait = len(hist['val_loss']) - hist['val_loss'].index(last_best) - 1
	else:
		last_best = np.Inf
		wait = 0	
	time_stamp = str(time.localtime()[1:5]).replace(',','').replace(' ','-')[1:-1] # month-day-hour-minute
	model_name = 'gen_model-{}'.format(time_stamp)	
	CHP = fwc.ModelCheckpoint('best_'+model_name.split('-')[0]+'.h5',
							  monitor='val_loss', 
							  mode='min',
							  save_best_only=True,
							  previous_best=last_best, 
							  verbose=1)										
	ES = fwc.EarlyStopping(monitor='val_loss', 
						   mode='min',
						   wait=wait, 
						   baseline=last_best,
					       verbose=1, 
						   patience=10)
	TB = callbacks.TensorBoard(log_dir='./logs/{}'.format(model_name)
							   )


	# Train model on dataset
	cb_list = [ES, CHP]
	path = '/data/andrej'
	for dataset_n in range(0,20):
		
		load_paths = [path+'/gen_train_x_{}.npy'.format(dataset_n), path+'/gen_train_y_{}.npy'.format(dataset_n)]
		print(load_paths)
		trainx = np.load(load_paths[0])#.astype('float32')/255   
		trainy = np.load(load_paths[1]) 
		
		epoch_size = 20
		n_epochs = int(trainx.shape[0]/epoch_size/16)
		for i in range(n_epochs):
			print('({}/{}){}'.format(i+1, n_epochs, dataset_n+1))
			if hist['val_loss']:
				last_best = min(hist['val_loss'])
				wait = len(hist['val_loss']) - hist['val_loss'].index(last_best) - 1
			else:
				last_best = np.Inf
				wait = 0
			CHP = fwc.ModelCheckpoint('best_'+model_name.split('-')[0]+'.h5',
									  monitor='val_loss', 
									  mode='min',
									  save_best_only=True,
									  previous_best=last_best, 
									  verbose=1)
			ES = fwc.EarlyStopping(monitor='val_loss', 
								   mode='min',
								   wait=wait, 
								   baseline=last_best,
							       verbose=1, 
								   patience=10)
			x_data = trainx[i*16*epoch_size:(i+1)*16*epoch_size,:,:,:].astype('float32')/255
			y_data = trainy[i*16*epoch_size:(i+1)*16*epoch_size,:,:,:].astype('float32')
			history = model.fit(x=x_data, y=y_data, batch_size=16, epochs=5, verbose=1, validation_data = (valx, valy), shuffle=True, callbacks=cb_list)
			#print(history.history)
			for i in range(len(history.history['loss'])):
				hist['loss'].append(history.history['loss'][i])
				hist['val_loss'].append(history.history['val_loss'][i])
			
			#print(hist)
			#print(K.get_value(model.optimizer.lr))
		#print(hist)
		fn.saveHist('./best_gen_hist.json', hist)


	# Check training
	#x_data = np.load('test_x_data.npy')/255#[0:1,:,:,:]
	#y_true = np.load('test_y_data.npy')#[0:1,:,:,:]
	#history = model.fit(x=x_data, y=y_true, batch_size=1, epochs=50, verbose=1, validation_data = (x_data, y_true), shuffle=False, callbacks=cb_list)	
	
	# Training on generator
	'''
	cb_list = [ES, CHP, TB]
	history = model.fit_generator(generator = training_generator,
								  validation_data = (valx, valy),
								  callbacks = cb_list,
								  verbose = 1,
								  epochs = 20,
								  use_multiprocessing = True,
								  workers = 7)

	#print(history.history)
	'''		
	




