import functions as fn
import numpy as np
import os

dirname = os.path.dirname(os.path.abspath(__file__))
data_dir= os.path.join(dirname, 'data')

train_pics_dir=os.path.join(data_dir, 'train_pics')
train_annot_dir=os.path.join(data_dir, 'train_annot')

eval_pics_dir=os.path.join(data_dir, 'eval_pics')
eval_annot_dir=os.path.join(data_dir, 'eval_annot')


if False: anchors, X=(fn.anchors_from_data(4, train_pics_dir, train_annot_dir))

x_train_data, y_train_data=fn.prep_pics( img_folder=train_pics_dir,  annot_dir=train_annot_dir,  stop=True, stopval=100)
np.save(os.path.join(data_dir,'x_train_data'),x_train_data)
np.save(os.path.join(data_dir,'y_train_data'),y_train_data)


x_eval_data, y_eval_data=fn.prep_pics( img_folder=eval_pics_dir,  annot_dir=eval_annot_dir,  stop=False)
np.save(os.path.join(data_dir,'x_eval_data'),x_eval_data)
np.save(os.path.join(data_dir,'y_eval_data'),y_eval_data)