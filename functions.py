import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
import cv2             # working with, mainly resizing, images
from tqdm import tqdm
from lxml import etree
import xml.etree.ElementTree as ET
import random
import os                  # dealing with directories
import numpy as np
import numpy.ma as ma
import tensorflow as tf

from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras import models
from keras import layers
from keras.layers import Input
from keras.utils.generic_utils import get_custom_objects





def prep_pics(img_folder='', annot_dir='', ref=(416,416), stop=True, stopval=100):
    if stop:
        num=stopval
        x_data=np.zeros((num,416, 416, 3))
        y_data=np.zeros((num,13,13,5))
        
    else:
        x_data=np.zeros((len(os.listdir(img_folder)),416, 416, 3))
        y_data=np.zeros((len(os.listdir(img_folder)),13,13,5))
        
    n=0
    for img in tqdm(os.listdir(img_folder)):
        n=n+1
        if stop and n>num:
            break
        truth= create_truth_arr(conv_bb_coord(rescale_bb(*read_xml(img, annot_dir))))
        path = os.path.join(img_folder,img)
        img = cv2.imread(path)
        img = cv2.resize(img, ref)
        x_data[n-1,...]=np.array(img)
        y_data[n-1,...]=truth
    return x_data, y_data



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

def custom_loss(y_true, y_pred, no_object_scale=0.5, bb_scale=5., object_scale=5.):
    #shape = y_true.shape #(13,13,5)
    
    #no object loss 
    no_objects_mask = ma.masked_equal(y_true, 0).mask
    no_object_loss = K.sum((0 - (y_pred*no_objects_mask)[:,:,0])**2)
    
    #object loss 
    object_loss = K.sum((1 - (y_pred * ~no_objects_mask)[:,:,0])**2)
    
    # loss from bounding boxes
    bb_loss= K.sum((y_true[:,:,1:] - (y_pred * ~no_objects_mask)[:,:,1:])**2)

    loss= no_object_scale * no_object_loss  +  bb_scale * bb_loss  +  object_loss*object_scale
    #K.print(loss)      
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
    #print()
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
                
        truth[a,b,0:]= 1,*bb[i:i+4]
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


