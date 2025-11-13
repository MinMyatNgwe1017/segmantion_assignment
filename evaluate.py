from torch.utils.data import DataLoader
import os 
import torch 
import pandas as pd 
import matplotlib.pyplot as plt
from  args import get_args
from trainer import validate_model
from dataset import BoneDataSet
from model import UNetLext
import cv2
import numpy as np
args=get_args()
test_set=pd.read_csv(os.path.join(args.csv_dir,"test.csv"))

fig,ax=plt.subplots(5,2,figsize=(10,15))
model=UNetLext()
weight=torch.load("./model/best_model.pth")
model.load_state_dict(weight)
model.eval()
def preprocess_image(path):
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    image=np.expand_dims(image,axis=0)
    image=(torch.tensor(image).permute(0,3,1,2)/255).float()
    return image 
sample_data_indice=test_set.sample(5).index

with torch.no_grad():
    for i,j in enumerate(sample_data_indice):
        image=preprocess_image(test_set.xrays[j])
        mask = model(image)
        mask = mask.detach().cpu().numpy()
            
        ax[i][1].imshow(mask[0,0], cmap='gray')
        ax[i][1].set_title("predicted image")
        ax[i][1].axis("off")
        ax[i][0].axis("off")

        ax[i][0].imshow(np.transpose(image[0],(1,2,0)))
        ax[i][0].set_title("Original image")

plt.tight_layout()
if not os.path.exists("./evaluate"):
    os.makedirs("./evaluate")
plt.savefig("./evaluate/plot.png")
plt.show()
