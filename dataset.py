from torch.utils.data import DataLoader,Dataset 

import  cv2 
import torch 
import numpy as np
class BoneDataSet(Dataset):
    def __init__(self,df,resize=224):
        self.df=df 
        self.size=resize
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,index):
        
        sample=self.df.iloc[index,:]
        image=sample.xrays
        mask=sample.masks
        image_array=cv2.imread(image)
        image_array_rbg=cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
        mask_array=cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
        
        image_array_rbg=cv2.resize(image_array_rbg,(self.size,self.size))
        mask_array=cv2.resize(mask_array,(self.size,self.size))
        mask_array=np.expand_dims(mask_array,axis=-1)

        image=(torch.tensor(image_array_rbg).permute(2,0,1)/255).float()
        mask=(torch.tensor(mask_array).permute(2,0,1)/255).float()
        return image,mask
        
        