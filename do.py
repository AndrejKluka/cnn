"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""
import sys
import os
import functions as fn
import numpy as np

''' 
Functions to execute from server bash terminal for convenience

functions=['list_data',						# 0
		   'clear_data',					# 1
		   'create_pretrain_data',			# 2
		   'create_pretrain_data_sample',	# 3
		   'count_workers',					# 4
		   'gen_val_data',					# 5
		   'clear logs',					# 6
		   'gen_train_data',				# 7
		   'get_train_data_in_data']		# 8
'''


def do_py():

	def get_bcg_plane_names():
		try:
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
			return bcg_names, plane_names
		except Exception as e:
			print('Error: ',e)	

	def gen_data(gen_params, n, save_paths=[]):
		try:	
			val_generator = fn.DataGenerator( **gen_params)
			x_data = np.empty((n*16,*(416,416,3)),dtype='uint8')
			y_data = np.empty((n*16, *(13,13,5) ),dtype='float32')
			for i in range(n):
				x, y = val_generator[i]
				print ('({}/{})'.format(i+1,n),x[:5,0,0,0])
				x_data[i*16:(i+1)*16,:,:,:] = (x*255).astype('uint8')
				y_data[i*16:(i+1)*16,:,:,:] = y.astype('float32')
			print('saving')	
			np.save(save_paths[0], x_data)   
			np.save(save_paths[1], y_data)   
		except Exception as e:
			print('Error: ',e)	
		
	print('**inputs**')
	for i in range(len(sys.argv)):
		print(sys.argv[i]) # prints vari
	print('**********')	
	functions=['list_data',						# 0
			   'clear_data',					# 1
			   'create_pretrain_data',			# 2
			   'create_pretrain_data_sample',	# 3
			   'count_workers',					# 4
			   'gen_val_data',					# 5
			   'clear logs',					# 6
			   'gen_train_data',				# 7
			   'get_train_data_in_data']		# 8
	
	try:
		if sys.argv[1]==functions[0] or sys.argv[1]==str(functions.index(functions[0])):
			path = '/data/andrej'
			files = os.listdir(path)
			for file in files:	print(file)
	
		
		
		if sys.argv[1]==functions[1] or sys.argv[1]==str(functions.index(functions[1])):
			path = '/data/andrej'
			files = os.listdir(path)
			for file in files:
				try:
					if file.split('.')[1]=='npy':
						print(path+'/'+file+'   is removed')
						os.remove(path+'/'+file)
				except Exception as e:
					print('Error: ',e)
	
	
	
		if sys.argv[1]==functions[2] or sys.argv[1]==str(functions.index(functions[2])):
			try:
				fn.prep_pretrain_pics(data_dir='/data/andrej/train2017',save_dir='/data/andrej/', \
									  ref=(416,416), stop=False, startval=0, stopval=100, save=True, load=False)
			except Exception as e:
				print('Error: ',e)
				
	

			
		if sys.argv[1]==functions[3] or sys.argv[1]==str(functions.index(functions[3])):
			try:
				fn.prep_pretrain_pics(data_dir='/data/andrej/train2017',save_dir='/home/andrej/', \
									  ref=(416,416), stop=True, startval=0, stopval=100, save=True, load=False)
			except Exception as e:
				print('Error: ',e)	



		if sys.argv[1]==functions[4] or sys.argv[1]==str(functions.index(functions[4])):
			try:
				import multiprocessing
				print(multiprocessing.cpu_count())
			except Exception as e:
				print('Error: ',e)	



		if sys.argv[1]==functions[5] or sys.argv[1]==str(functions.index(functions[5])):
			bcg_names, plane_names = get_bcg_plane_names()
			train_params = {'bcg_paths': bcg_names[:int(len(bcg_names)*0.05)],
						    'plane_paths': plane_names[:int(len(bcg_names)*0.05)],
							'steps_per_epoch': 5,
							'batch_size': 16,
							'ref_in': (416,416,3),
							'ref_out': (13,13,5)}			
			gen_data(train_params, 10, save_paths=['./gen_val_x.npy','./gen_val_y.npy'])
	
		
				
		if sys.argv[1]==functions[6] or sys.argv[1]==str(functions.index(functions[6])):
			import shutil
			path = '/home/andrej/logs'
			files = os.listdir(path)
			for file in files:
				try:
					#if file.split('.')[1]=='npy':
					print(path+'/'+file+'   is removed')
					shutil.rmtree(path+'/'+file)
				except Exception as e:
					print('Error: ',e)
	

				
		if sys.argv[1]==functions[7] or sys.argv[1]==str(functions.index(functions[7])):
			
			bcg_names, plane_names = get_bcg_plane_names()
			
			train_params = {'bcg_paths': bcg_names[int(len(bcg_names)*0.05):],
						    'plane_paths': plane_names[int(len(bcg_names)*0.05):],
							'steps_per_epoch': 5,
							'batch_size': 16,
							'ref_in': (416,416,3),
							'ref_out': (13,13,5)}			
			
			gen_data(train_params, 100, save_paths=['./gen_train_x.npy','./gen_train_y.npy'])



		if sys.argv[1]==functions[8] or sys.argv[1]==str(functions.index(functions[8])):
			
			bcg_names, plane_names = get_bcg_plane_names()
			
			train_params = {'bcg_paths': bcg_names[int(len(bcg_names)*0.05):],
						    'plane_paths': plane_names[int(len(bcg_names)*0.05):],
							'steps_per_epoch': 5,
							'batch_size': 16,
							'ref_in': (416,416,3),
							'ref_out': (13,13,5)}	
			
			path = '/data/andrej'
			for i in range(10,20):
				save_paths = [path+'/gen_train_x_{}.npy'.format(i), path+'/gen_train_y_{}.npy'.format(i)]
				print(save_paths)
				gen_data(train_params, 200, save_paths= save_paths)
			
			
	except Exception as e:
		print('Error: ',e)
	
	print('\npossible functions below:')
	for i in range(len(functions)):
		print(str(i)+': ',functions[i])	



if __name__ == '__main__':
	do_py()
















