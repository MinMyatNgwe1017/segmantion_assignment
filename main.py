from args import get_args
from trainer import train_model
import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import DataLoader
import cv2
import os 
import pandas as pd 
from model import UNetLext
from trainer import train_model,validate_model
from dataset import BoneDataSet

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def worker_init(worker_id):
    torch.manual_seed(42+worker_id)


def main():
    args=get_args()
    train_set=pd.read_csv(os.path.join(args.csv_dir,"train.csv"))
    val_set=pd.read_csv(os.path.join(args.csv_dir,"val.csv"))
    
    train_data=BoneDataSet(train_set)
    val_data=BoneDataSet(val_set)
    
    train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=3,worker_init_fn=worker_init)
    validate_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=False)
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=UNetLext(input_channels=3,output_channels=1)
    model.to(device)

    history=train_model(model=model,train_loader=train_loader,val_loader=validate_loader,device=device)
    df = pd.DataFrame(history)
    if not os.path.exists("training_plot"):
        os.mkdir("training_plot")
            
    plt.figure(figsize=(10, 6))
    df.plot()
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.savefig("training_plot/training_plot.png")
    plt.close()
if __name__=="__main__":
    main()
