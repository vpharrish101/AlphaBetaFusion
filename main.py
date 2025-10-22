import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt

from tqdm import tqdm
from IPython.display import display, clear_output
from transformers import AutoTokenizer,AutoModel
from torch.amp import autocast,GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR

import contactmap

from config import DEVICE,CLASSES,BATCH_SIZE,EPOCH,VIT_HIDDEN
from DatasetLoader import DatasetLoader
from engine import ViT_BERT_model

def main():
    Dataset=DatasetLoader()
    scaler=GradScaler()
    device=torch.device(DEVICE)

    #run contactmap.py to generate the dataset <contact_maps,FASTA seq>

    ViT_model=timm.create_model("vit_tiny_patch16_224",pretrained=True)
    BERT_tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    BERT_model=AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")


    for param in ViT_model.parameters():
        param.requires_grad=False
    for i in range(7,12):
        for param in ViT_model.blocks[i].parameters():
            param.requires_grad=True


    num_layers=len(BERT_model.encoder.layer) 
    for layer in BERT_model.encoder.layer[num_layers-2:]: 
        for param in layer.parameters():
            param.requires_grad=True


    ViT_model.to(device)
    BERT_model.to(device)


    preload_weights=False
    if preload_weights==True:
        ckpt_path = r"D:\Python311\ViT-BERT\CATH\ViT pet project\ViT_BERT_model_partial.pth"
        state_dict = torch.load(ckpt_path, map_location=device)
        learner=ViT_BERT_model(ViT_model,BERT_tokenizer,BERT_model,CLASSES,VIT_HIDDEN,loss_fn,device)
        learner.load_state_dict(state_dict,strict=False)
        learner.to(device)

    learner=torch.compile(learner,dynamic=False,backend="aot_eager")
    low_lr_params=[]
    for i in range(7,12):
        low_lr_params.extend(ViT_model.blocks[i].parameters())
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([2,2,2,1],dtype=torch.float).to(device))
    
    base_lr=1.5e-3
    low_lr=6e-4
    high_lr=1.8e-3
    for layer in BERT_model.encoder.layer[num_layers-4:]:
        low_lr_params.extend(layer.parameters())

    high_lr_params=[]
    high_lr_params.extend(learner.ViT_model.head.parameters()) 
    high_lr_params.extend(learner.FusionLayer.parameters())

    optimizer=torch.optim.AdamW([
            {'params':filter(lambda p:p.requires_grad,low_lr_params),'lr':low_lr,'weight_decay':0.03}, 
            {'params':filter(lambda p:p.requires_grad,high_lr_params),'lr':high_lr,'weight_decay':0.03}],
        lr=base_lr,weight_decay=0.05,foreach=True)
    
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCH,eta_min=1e-6)
    warmup=LinearLR(optimizer,start_factor=0.2,total_iters=5)
    scheduler=SequentialLR(optimizer,schedulers=[warmup,scheduler],milestones=[5])
    

    train_data=(Dataset.TrainSplit(BATCH_SIZE))
    accumulation_steps=1


    plt.ion()
    fig,ax=plt.subplots()
    ax.set_title("Training Accuracy (Live)")
    ax.set_xlabel("Effective Batch")
    ax.set_ylabel("Accuracy")
    line_train,=ax.plot([],[],'b-',label='Train Accuracy')
    ax.legend()
    plt.show(block=False)

    optimizer.zero_grad() 
    accum_preds,accum_labels=[],[]

    try: 
        for epoch in range(EPOCH):
            learner.train()
            epoch_loss=0
            loop=tqdm(train_data,desc=f"Epoch {epoch+1}",leave=False)
            train_acc_history=[]

            for batch_idx,(image,labels,sequence) in enumerate(loop):
                image=image.to(device)
                labels=labels.to(device).long()

                with autocast(device_type="cuda",dtype=torch.float16):
                    loss,logits=learner.forward(image,labels,sequence)
                    loss=loss/accumulation_steps 
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(learner.parameters(),max_norm=1.0)
                
                accum_preds.append(logits.detach().cpu())
                accum_labels.append(labels.detach().cpu())

                if (batch_idx +1)%accumulation_steps==0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    logits_concat=torch.cat(accum_preds,dim=0)
                    labels_concat=torch.cat(accum_labels,dim=0)
                    batch_acc=(torch.argmax(logits_concat,dim=1)==labels_concat).float().mean().item()
                    train_acc_history.append(batch_acc)
                    if len(train_acc_history)>1000:
                        train_acc_history=train_acc_history[-1000:] 
                    accum_preds,accum_labels=[],[]

                    clear_output(wait=True) 
                    line_train.set_data(range(len(train_acc_history)),train_acc_history)
                    ax.relim()
                    ax.autoscale_view()
                    display(fig)

                epoch_loss+=loss.item()
                loop.set_postfix(loss=loss.detach().item())

            print(f"Epoch {epoch+1} avg loss: {epoch_loss/len(train_data):.4f}")
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user, saving current weights")
        torch.save(learner.state_dict(),"ViT_BERT_model_partial.pth")

    plt.ioff()
    plt.show()
    
    test_loader=Dataset.TestSplit(BATCH_SIZE)
    all_correct=0
    all_total=0
    with torch.no_grad():
        for test_images,test_labels,test_seq in test_loader:
            test_images=test_images.to(device)
            test_labels=test_labels.to(device).long()
            _,test_logits=learner.forward(test_images,test_labels,test_seq)
            test_preds=torch.argmax(test_logits,dim=1)
            all_correct+=(test_preds==test_labels).sum().item()
            all_total+=test_labels.size(0)

    test_acc=all_correct/all_total
    print(f"Test Accuracy: {test_acc:.4f}")
    plt.ioff()
    plt.show()
    learner.eval()
    torch.save(learner.state_dict(),"ViT_BERT_model_weights.pth")


if __name__=="__main__":
    main()