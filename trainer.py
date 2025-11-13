from args import get_args
import torch 
from utils import dice_loss_from_logits,dice_score_from_logits
import os 
def train_model(model,train_loader,val_loader,device,early_stop=True):  
    history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    args=get_args()
    critersion=torch.nn.BCEWithLogitsLoss()
    best_score=0
    count=0
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    for epoch in range(args.epochs):
        training_loss=0
        training_score=0
        model.train()
        for image,mask in train_loader:
            image=image.to(device)
            mask=mask.to(device)
            optimizer.zero_grad()
            output=model(image)
        
            loss_bce=critersion(output,mask)
            
            loss_dice=dice_loss_from_logits(output,mask)
            
            loss=loss_bce+loss_dice

            loss.backward()
            
            optimizer.step()
            
            training_loss+=loss.item()
            score=dice_score_from_logits(output,mask)
            training_score+=score
            
        training_loss/=len(train_loader)
        training_score/=len(train_loader)
        
        val_loss,val_score=validate_model(model,val_loader,device,critersion)
        
        print(
            f"epoch {epoch+1}/{args.epochs}\t training loss {training_loss} acc {training_score}\t val loss {val_loss} val score{val_score}"
        )
        history["train_loss"].append(training_loss)
        history["train_acc"].append(training_score)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_score)

        if early_stop:
            if best_score<val_score:
                if not os.path.exists("./model"):
                    os.mkdir("./model")
                torch.save(model.state_dict(),"./model/best_model.pth")

                count=0
                best_score=val_score
            else:
                count+=1
            if count>=args.patience:
                break 
    return history

def validate_model(model,val_loader,device,loss_fn):
    model.eval()
    val_loss=0.0
    val_score=0.0
    with torch.no_grad():
        for image,mask in val_loader:
            image=image.to(device)
            mask=mask.to(device)
            output=model(image)
            loss_bce=loss_fn(output,mask)
            
            loss_dice=dice_loss_from_logits(output,mask)
            loss=loss_bce+loss_dice
            
            val_loss+=loss.item()
            val_score+=dice_score_from_logits(output,mask)
            
            
        val_loss/=len(val_loader)
        val_score/=len(val_loader)
    return val_loss,val_score