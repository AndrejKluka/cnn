"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


# !!! code needs older spyder with python console to work
#conda uninstall spyder

#conda install spyder=3.1.4


file__path=os.path.dirname(__file__)
#all_xml(reqcat,trainorval,savedir,file__path)

savedir = 'all_backgrounds'
if not os.path.isdir(savedir):
    os.mkdir(savedir)

background_folders = os.listdir('dataset')

def onkeypress(event):
	global horizoness, horizon, imname, n, fig
	
	if event.key == 'f':  #the whole picture is sky(above horizon)
		horizoness.append([imname, 'above'])
		fig.canvas.stop_event_loop()
		plt.close()
		
	if event.key == 'v':  #the whole picture land/water(below horizon)
		horizoness.append([imname, 'below'])
		fig.canvas.stop_event_loop()
		plt.close()
		
	if event.key == 'g':  #the pictur is well annotated by mouse
		if len(horizon) > 1:
			horizoness.append([imname, 'defined', horizon])
			fig.canvas.stop_event_loop()
			plt.close()
		else:
			print('put in more points')
		
	if event.key == 'b':  #show last picture
		del horizoness[-1]
		n=n-2
		fig.canvas.stop_event_loop()
		plt.close()

def onmousepress(event):
	global horizon
	horizon.append([event.xdata,event.ydata])  #list of x-z=y coords



def sort_imgs( to_break=False, savedir = 'all_backgrounds', nstart=0):
	global n, horizoness, imname, fig, horizon

	print('press F if whole pic is above horizon \npress V if whole pic is below horizon \n B for going back \n G for good annotation ')
	horizoness=[]
	n = nstart
	folder_path=os.path.join(os.path.dirname(__file__),savedir)
	target_folder = os.listdir(folder_path)
	while n<len(target_folder):
		if n<0 : n = 0
		horizon=[]
		imname = target_folder[n]
		impath = os.path.join(folder_path, imname)
		# set windows size and read plane path 
		fig, ax = plt.subplots(1)
		mngr = plt.get_current_fig_manager()
		mngr.window.setGeometry(250, 100, 1280, 824)
		image = cv2.imread(impath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)		
		ax.imshow(image)
		plt.connect('key_press_event', onkeypress)
		plt.connect('button_press_event', onmousepress)
		plt.show()	
		fig.canvas.start_event_loop()		
		
						
		
		
		n += 1
		if n % 25 == 0:
			print(n)
			np.save('horizons',horizoness)
		if to_break:            
			if n>5:
				break

	return horizoness


#listecek = sort_imgs(nstart=0, to_break=False)
#np.save('horizons',listecek)
#a=np.load('horizons.npy')
#for i in listecek: print(i)




