"""
Created on Mon Aug 26 20:08:06 2019

@author: Andrej Kluka
"""
import coco
import os
import shutil

trainorval='train'

# Separate all pics with plane category ID=5 from COCO dataset

annotation_file=os.path.join(os.path.join(os.path.dirname(__file__),'annotations'),'instances_{}2017.json'.format(trainorval))
co=coco.COCO(annotation_file)
print(co.createIndex())

plane_ids=co.getImgIds(catIds=[5])

#print(co.cats)
savedir='{}{}'.format(trainorval,co.cats[5]['name'])
savedir_path=os.path.join(os.path.dirname(__file__),savedir)

print(len(plane_ids))
plane_imgs=co.loadImgs(ids=plane_ids)

if not os.path.isdir(savedir):
    os.mkdir(savedir)

img_path=os.path.join(os.path.dirname(__file__),'{}2017'.format(trainorval))


for plane in plane_imgs:
    shutil.copy(os.path.join(img_path,plane['file_name']), savedir_path)



'''
src_dir = "your/source/dir"
dst_dir = "your/destination/dir"
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    shutil.copy(jpgfile, dst_dir)
'''    
    
    
    
    