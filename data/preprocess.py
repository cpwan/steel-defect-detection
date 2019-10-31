'''#Preprocess masks'''
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
# folder should contain:
# mask_path
# data_folder
# df_path


mask_path='mask' 
data_folder='img'
df_path='train.csv'



try:
    # Create target Directory
    os.mkdir(mask_path)
    print("Directory " , mask_path ,  " Created ") 
except FileExistsError:
    print("Directory " , mask_path ,  " already exists")

try:
    # Create target Directory
    os.mkdir(mask_path+'/train')
    print("Directory " , mask_path+'/train' ,  " Created ") 
except FileExistsError:
    print("Directory " , mask_path+'/train' ,  " already exists")

try:
    # Create target Directory
    os.mkdir(mask_path+'/test')
    print("Directory " , mask_path+'/test' ,  " Created ") 
except FileExistsError:
    print("Directory " , mask_path+'/test' ,  " already exists")


 
def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`
    ---
    Reference https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88    '''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
#     print('fname',fname)
    return fname, masks

def oneHot2label(mask):
  ''' label:
         no defect: 4
         defect 0: 0
         defect 1: 1
         defect 2: 2
         defect 3: 3
  '''
  no_defect_label=255
  out_array=np.zeros(mask[:,:,0].shape, dtype=np.uint8)+no_defect_label
  for i in range(mask.shape[2]):
    out_array[mask[:,:,i]==1] = i
  no_defect = (out_array == no_defect_label).all()
  return Image.fromarray(out_array,'L'), no_defect

'''Returns dataloader for the model training'''
df = pd.read_csv(df_path)
df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
df['ClassId'] = df['ClassId'].astype(int)
df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
df['defects'] = df.count(axis=1)
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["defects"], random_state=69)


for row_id in range(len(train_df.index)):  
  fname,mask=make_mask(row_id, train_df)
  masked_image, no_defect=oneHot2label(mask)
  if no_defect:
    continue
  out_name=os.path.join(mask_path,'train',fname.split('.')[0]+'.png')
  masked_image.save(out_name)
for row_id in range(len(test_df.index)):  
  fname,mask=make_mask(row_id, test_df)
  masked_image, no_defect=oneHot2label(mask)
  out_name=os.path.join(mask_path,'test',fname.split('.')[0]+'.png')
  masked_image.save(out_name)
print('Done!')
