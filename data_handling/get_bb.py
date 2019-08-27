"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""

import coco
import os
import cv2
import numpy as np
from lxml import etree
import xml.etree.cElementTree as ET

#from generate_xml import write_xml

trainorval='val'
reqcat=5

annotation_file=os.path.join(os.path.join(os.path.dirname(__file__),'annotations'),'instances_{}2017.json'.format(trainorval))
co=coco.COCO(annotation_file)
print(co.createIndex())

plane_ids=co.getImgIds(catIds=[reqcat])
planeann_ids=co.getAnnIds(catIds=[reqcat])
planeann_ids2=co.getAnnIds(imgIds=plane_ids)

anns=co.loadAnns(ids=planeann_ids)
#co.showAnns(anns[0])
#anns[0]['bbox']
#anns[0]['image_id']
#anns[0]['segmentation']
#co.loadImgs(ids=[anns[0]['image_id']])[0]['file_name']
#300659
#for i in anns:print(i['image_id'])

for i in range(len(plane_ids)):
    if plane_ids[i]==300659 :
        print(i)
        n=i


img_dir=os.path.join(os.path.dirname(__file__),'{}2017'.format(trainorval))

def showim(n):
    annIds=co.getAnnIds(imgIds=plane_ids[n])
    img_name=co.imgs[plane_ids[n]]['file_name']
    image_file= os.path.join(img_dir,img_name)
    image = cv2.imread(image_file)
    for annId in annIds:
        if co.anns[annId]['category_id']==reqcat:
            bbox=co.anns[annId]['bbox']
            #for seg in anns[n]['segmentation']:
            seg=co.anns[annId]['segmentation'][0]
            poly = np.array(seg).reshape((int(len(seg)/2), 2))
            bbox[0]=np.min(poly[:,0])
            bbox[1]=np.min(poly[:,1])
            bbox[2]=np.max(poly[:,0])
            bbox[3]=np.max(poly[:,1])
            image = cv2.rectangle(image, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 255, 0), 3)
    cv2.imshow('Test image',image)

def all_xml(reqcat,trainorval,savedir):


def write_xml(folder, img, objects, tl, br, savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    image = cv2.imread(img.path)
    height, width, depth = image.shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = img.name
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
    save_path = os.path.join(savedir, img.name.replace('png', 'xml'))
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


showim(n)
#co.anns[57620]
'''

anns[0].keys()
Out[18]: dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
'''
