import cv2
import os
import numpy as np
import math
import yaml
import glob
import tables
import numpy as np
from progress.bar import Bar
import resource






def find_im_path(path):
    for root,dirs,files in os.walk(path):
        for name in sorted(files):
            if name.endswith('g'):
                return root+'/'+name

def write_train_data_to_file(path,earray,data_ind,inds,count,ext,bar):
    fl=0
    for c,name in enumerate(sorted(os.listdir(path))):
        if os.path.isdir(path+'/'+name):

            fl=1
            if len(inds)==0:
                data_ind.append([])
            else:
                ind=''.join(['['+str(i)+']' for i in inds])
                eval( 'data_ind'+ind+'.append([])')
            _,earray,data_ind,_,count,bar=write_train_data_to_file(path+'/'+name,earray,data_ind,inds+[c],count,ext,bar)

    if not fl:
        for name in sorted(os.listdir(path)):
            if name.endswith(ext):
                ind=''.join(['['+str(i)+']' for i in inds])
                eval( 'data_ind'+ind+'.append([count])')
                earray.append(np.expand_dims(cv2.imread(path+'/'+name,0),axis=0))


                count+=1
                bar.next()
    return '',earray,data_ind,0,count,bar



def Save_Action_Gesture():
    action_train_data_path="/media/vassilis/Data/Thesis/Datasets/Cambridge_Action_Gesture_Dataset"
    ext='.jpg'
    filename = 'TrainData/Actions.h5'
    im_path=find_im_path(action_train_data_path)
    im=cv2.imread(im_path,0)
    imsize=im.shape
    cpt = float(sum( sum(map(lambda x: x.endswith(ext),files))  for r, d, files in os.walk(action_train_data_path)))
    bar = Bar('Reading Action Data', max=cpt)

    f = tables.open_file(filename, mode='w')
    atom = tables.Atom.from_dtype(im.dtype)
    array_c = f.create_earray(f.root, 'array_c', atom, (0,)+imsize)
    _,earray,data_inds,_,_,_=write_train_data_to_file(action_train_data_path,array_c,[],[],0,ext,bar)
    f.close()
    np.save('TrainData/Action_inds',data_inds)

'''
#Check if the data was collected correctly
i=62000
data_inds=np.load('TrainData/Action_inds.npy').tolist()
def find_3d_indices(data_inds,i):
 for c1,col1 in enumerate(data_inds):
  for c2,col2 in enumerate(col1):
   for c3,col3 in enumerate(col2):
    for c4,el in enumerate(col3):
     if el==[i]:
      return c1,c2,c3,c4

ind=find_3d_indices(data_inds,i)

print ind
hdf5_path = "TrainData/Actions.h5"


hdf5 = tables.open_file(hdf5_path, mode='r')#root_uep='/array_c')
im=hdf5.root.array_c[i,:,:,:]
print im.shape
cv2.imshow("im",im)
cv2.waitKey(0)
'''
