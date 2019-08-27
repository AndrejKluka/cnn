"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""
import os
import matplotlib.pyplot as plt
import cv2


# !!! code needs older spyder with python console to work
#conda uninstall spyder

#conda install spyder=3.1.4


file__path=os.path.dirname(__file__)
#all_xml(reqcat,trainorval,savedir,file__path)

savedir = 'all_backgrounds'
if not os.path.isdir(savedir):
    os.mkdir(savedir)



def onkeypress(event):
	global good_background, okness, n , fig, name_n, previous
	if event.key == 'a':
		okness = 'ok'
		fig.canvas.stop_event_loop()
		plt.close()
	if event.key == 'l':
		okness = 'bad'
		fig.canvas.stop_event_loop()
		plt.close()
	if event.key == 'b':
		okness = 'nah'
		if n==0:
			print('cant go back')
		else:
			n=n-2
			if previous=='ok':
				name_n -= 1
			fig.canvas.stop_event_loop()
			plt.close()


def sort_imgs( to_break=False, savedir='all_backgrounds'):
	global okness, n, fig, name_n, previous
	name_n = 0
	try:
		check_savedir = os.listdir(os.path.join(os.path.dirname(__file__),savedir))[-1]
		m = int(check_savedir.split('.')[0])
	except:
		m = 0
	file__path=os.path.join(os.path.dirname(__file__),'dataset')
	background_folders = os.listdir(file__path)
	for folder in background_folders:
		folder_path=os.path.join(file__path,folder)
		n = 0	
		target_folder = os.listdir(folder_path)
		while n<len(target_folder):
			previous='nothing'
			imname = target_folder[n]
			impath = os.path.join(folder_path, imname)
			
			# set windows size and read plane path 
			try:
				fig, ax = plt.subplots(1)
				mngr = plt.get_current_fig_manager()
				mngr.window.setGeometry(250, 100, 1280, 824)
				image = cv2.imread(impath)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)		
				ax.imshow(image)
				plt.connect('key_press_event', onkeypress)
				plt.show()	
				fig.canvas.start_event_loop()
				#print(name_n+m+1)
				if okness == 'ok':
					previous='ok'
					plane_name = '{:06}.png'.format(name_n+m+1)
					
					os.rename(impath, os.path.join(os.path.dirname(__file__),savedir, plane_name))
					name_n += 1
				
				elif previous=='ok' and okness == 'nah':
					os.rename(os.path.join(os.path.dirname(__file__),savedir,'{:06}.png'.format(name_n+1+m)) , os.path.join(folder_path,  target_folder[n+1]))
				n += 1
				if to_break:            
					if n>5:
						break
			except Exception as e:
				n += 1
				print ("idk man", e)



sort_imgs(to_break=False, savedir='all_backgrounds')

