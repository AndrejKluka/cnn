"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""
import os
import matplotlib.pyplot as plt
import cv2
import coco
import numpy as np
from lxml import etree
import xml.etree.cElementTree as ET

# !!! code needs older spyder with python console to work
#conda uninstall spyder
#conda install spyder=3.1.4

trainorval='train'
reqcat=5
savedir='annot_{}'.format(trainorval)
file__path=os.path.dirname(__file__)
#all_xml(reqcat,trainorval,savedir,file__path)


def onkeypress(event):
    global good_full_plane_ids
    global plane_name
    global to_loop
    global n
    if event.key == 'a':
        #print('plane is ok')
        good_full_plane_ids.append('ok '+plane_name)
        #print(good_full_plane_ids)
        #write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
        plt.close()
    if event.key == 'l':
        good_full_plane_ids.append('bad '+plane_name)
        #print('plane is not ok')
        #write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
        #image = None
        plt.close()
    if event.key == 'b':
        del good_full_plane_ids[-1]
        #print('going back')
        n=n-2
        #write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
        #image = None
        plt.close()


def sort_imgs(annotation_folder,  reqcat=5,  to_break=False):   #reqcat = 5 for airplanes
    #get airplane paths
    global plane_name
    global n
    co=coco.COCO(annotation_folder)
    plane_ids=co.getImgIds(catIds=[reqcat])
    n=0
    
    #iterate over airplane IDs
    while n< len(plane_ids):
    #for plane_id in plane_ids:
        plane_img=co.loadImgs(ids=plane_ids[n])[0]
        plane_name= plane_img['file_name']   # '000000221184.jpg'
        img_path=os.path.join(img_dir,plane_name)
    
        # set windows size and read plane path 
        fig, ax = plt.subplots(1)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(250, 100, 1280, 824)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #get contour
        annIds=co.getAnnIds(imgIds=plane_ids[n])
        for annId in annIds:
            if co.anns[annId]['category_id']==reqcat:
                try:
                    seg=co.anns[annId]['segmentation'][0]
                    poly = np.array(seg).reshape((int(len(seg)/2),1, 2)).astype(int)  #arr(points,1,2)
                    image_con=image
                    image_con=cv2.drawContours(image_con, [poly],0,(0,255,0),3)
                except:print(co.imgs[plane_ids[n]]['file_name'],plane_ids[n],annId)
        #show images and connect event
        ax.imshow(image_con)
        plt.connect('key_press_event', onkeypress)
        plt.show()
        n+=1
        if n % 50 == 0:
            print(n)
            np.save('good_pic_name',good_full_plane_ids)
        if to_break:            
            if n>5:
                break



plane_name=None

img_folder='instances_{}2017.json'.format(trainorval)
annotation_folder=os.path.join(file__path,'annotations',img_folder)
img_dir=os.path.join(file__path,'{}2017'.format(trainorval))
good_full_plane_ids=[]
sort_imgs(annotation_folder,  reqcat=5,  to_break=False)

np.save('good_pic_name',good_full_plane_ids)

a=np.load('good_pic_name.npy')

print(n,len(a))

def write_xml(objects, tl, br, savedir,image_path,img_name,img_folder):
    image = cv2.imread(image_path)
    height, width, depth = image.shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text ='images'# img_folder.strip('.json')
    ET.SubElement(annotation, 'filename').text = img_name
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    for obj, topl, botr in zip(objects, tl, br):
        ob = ET.SubElement(annotation, 'object')
        ET.SubElement(ob, 'name').text = obj
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text = '0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(topl[0])
        ET.SubElement(bbox, 'ymin').text = str(topl[1])
        ET.SubElement(bbox, 'xmax').text = str(botr[0])
        ET.SubElement(bbox, 'ymax').text = str(botr[1])

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir, img_name.replace('jpg', 'xml'))
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


