import torch
import torch.nn as nn

class ViT_BERT_model(nn.Module):
    def __init__(self,ViT_model,BERT_tokenizer,BERT_model,num_classes,vit_hidden,loss_fn,device):
        super().__init__()
        self.device=device
        self.loss_fn=loss_fn
        self.ViT_model=ViT_model
        self.BERT_tokenizer=BERT_tokenizer
        self.BERT_model=BERT_model
    
        self.ViT_model.head=nn.Linear(ViT_model.head.in_features,vit_hidden,bias=True)
        self.ViT_model.to(self.device)

        self.FusionLayer=nn.Sequential(
            nn.Linear(vit_hidden+BERT_model.config.hidden_size,512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512,num_classes)
        ).to(device)
    
    def forward(self,image_tensor,labels,sequence):
        
        #ViT embeddings
        esm_ViT=self.ViT_model(image_tensor)

        #BERT embeddings
        seq=self.BERT_tokenizer(list(sequence),return_tensors="pt",padding=True,truncation=True,max_length=1024)
        encoding={k:v.to(self.device) for k, v in seq.items()}
        output=self.BERT_model(**encoding)
        esm_BERT=output.pooler_output

        #fusion
        fused_embeddings=torch.cat([esm_ViT,esm_BERT],dim=1)
        logits=self.FusionLayer(fused_embeddings)
        loss=self.loss_fn(logits,labels)
        #del fused_embeddings,esm_BERT,encoding,esm_ViT
        #torch.cuda.empty_cache()
        return loss,logits