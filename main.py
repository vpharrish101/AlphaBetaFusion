#package dependencies: -
import torch
import torch.nn as nn
import timm

from transformers import AutoTokenizer,AutoModel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast,GradScaler


#file_dependencies: -
import contactmap

from config import DEVICE,CLASSES,BATCH_SIZE,EPOCH,VIT_HIDDEN
from DatasetLoader import DatasetLoader
from engine import ViT_BERT_model




def __main__():
    Dataset=DatasetLoader()
    device=torch.device(DEVICE)

    #run this to generate contact maps
    #contactmap.main()                                                            

    ViT_model=timm.create_model("vit_small_patch16_224",pretrained=True)
    BERT_tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    BERT_model=AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    ViT_model.to(device)
    BERT_model.to(device)

    for param in ViT_model.parameters():
        param.requires_grad=False
    for i in range(9,12):
        for param in ViT_model.blocks[i].parameters():
            param.requires_grad=True

    num_layers=len(BERT_model.encoder.layer)  
    for layer in BERT_model.encoder.layer[num_layers-2:]: 
        for param in layer.parameters():
            param.requires_grad=True

    loss_fn=nn.CrossEntropyLoss()
    scaler=GradScaler()
    optimizer=torch.optim.AdamW(
        filter(lambda p:p.requires_grad,list(ViT_model.parameters())+list(BERT_model.parameters())),
        lr=1e-4,
        weight_decay=0.05)

    learner=ViT_BERT_model(ViT_model,BERT_tokenizer,BERT_model,CLASSES,VIT_HIDDEN,loss_fn,device)

    for x in range(EPOCH):
        learner.train()
        train_data=(Dataset.TrainSplit(BATCH_SIZE))

        for _,(image,labels,sequence) in enumerate(train_data, start=1):
            image=image.to(device)
            labels=labels.to(device).long()
            optimizer.zero_grad()
            
            with autocast(device_type="cuda",dtype=torch.float16):
                loss=learner.forward(image,labels,sequence)

            scaler.scale(loss).backward()
            scaler.step(optimizer)    
            scaler.update()
            print(f"Epoch {x+1}: loss={loss.item():.4f}")