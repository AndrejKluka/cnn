"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""
import os
import matplotlib.pyplot as plt
import cv2
import coco
import numpy as np

def onkeypress(event):
	global adjusted_good_cuts
	global plane_name
	global n
	global last_good
	if event.key == 'a':
		adjusted_good_cuts.append('ok '+plane_name)
		last_good=n
		plt.close()
	if event.key == 'l':
		adjusted_good_cuts.append('bad '+plane_name)
		plt.close()
	if event.key == 'b':
		n=last_good -1
		while len(adjusted_good_cuts)>n+1:
			del adjusted_good_cuts[-1]
		plt.close()

trainorval='train'
reqcat=5
savedir='cut_planes_{}'.format(trainorval)
if not os.path.isdir(savedir):
	os.mkdir(savedir)
file__path=os.path.dirname(__file__)
ann_folder='instances_{}2017.json'.format(trainorval)
annotation_folder=os.path.join(file__path,'annotations',ann_folder)
img_dir=os.path.join(file__path,'{}2017'.format(trainorval))

good_plane_names=np.load('good_plane_names.npy')
n=0

co=coco.COCO(annotation_folder)
plane_ids=co.getImgIds(catIds=[reqcat])
adjusted_good_cuts=[]

while n< len(good_plane_names):
	plane_name_plus=good_plane_names[n]
	
	plane_img=co.loadImgs(ids=plane_ids[n])[0]
	plane_name_one= plane_img['file_name']   # '000000221184.jpg'
	
	#print(n,adjusted_good_cuts)
	value, plane_name = plane_name_plus.split(' ')
	
	if value== 'ok' and plane_name_one==plane_name:		
		fig, ax = plt.subplots(1)
		img_path=os.path.join(img_dir,plane_name)
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		#get contour
		annIds=co.getAnnIds(imgIds=plane_ids[n])
		for annId in annIds:
			if co.anns[annId]['category_id']==reqcat:
				try:
					seg=co.anns[annId]['segmentation'][0]
					poly = np.array(seg).reshape((int(len(seg)/2),1, 2)).astype(int)  #arr(points,1,2)
				except:print(co.imgs[plane_ids[n]]['file_name'],plane_ids[n],annId)		
		
		#get the inside of contour
		mask = np.zeros_like(image)   #0 is black, 255 is white
		mask = cv2.drawContours(mask, [poly],0,(255,255,255),-1)  #creates white inside contour
		plane = np.zeros_like(image) # Extract out the object and place into output image
		plane[:,:,1]=plane[:,:,1]+254
		plane[mask == 255] = image[mask == 255]  #extracts OG pic where mask is white
					
		# Now crop outside of contour
		(x, y) = np.where(mask[:,:,0] == 255)
		(topx, topy) = (np.min(x), np.min(y))
		(bottomx, bottomy) = (np.max(x), np.max(y))
		plane = plane[topx:bottomx+1, topy:bottomy+1]
		#mask = mask[topx:bottomx+1, topy:bottomy+1]	
		
		ax.imshow(plane)
		plt.connect('key_press_event', onkeypress)
		plt.show()
		#save_place=os.path.join(savedir, plane_name)
		#plane= cv2.cvtColor(plane, cv2.COLOR_RGB2BGR)
		#cv2.imwrite(save_place, plane)
		
	elif value== 'bad' and plane_name_one==plane_name:
		adjusted_good_cuts.append('bad '+plane_name)
		
	elif not plane_name_one==plane_name:
		print(n,value,plane_name)
	n+=1
	if n % 50 == 0:
		print(n)
		np.save('adjusted_good_cuts',adjusted_good_cuts)
	#if n>10:
		#break



np.save('adjusted_good_cuts',adjusted_good_cuts)

a=np.load('adjusted_good_cuts.npy')

print(n,len(a))






