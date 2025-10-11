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

        self.FusionLayer=nn.Sequential(
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(vit_hidden+BERT_model.config.hidden_size,num_classes)
        )
    
    def forward(self,image_tensor,labels,sequence):
        
        #ViT embeddings
        esm_ViT=self.ViT_model(image_tensor)

        #BERT embeddings
        seq=self.BERT_tokenizer(list(sequence),return_tensors="pt",padding=True,truncation=True)
        encoding={k:v.to(self.device) for k, v in seq.items()}
        output=self.BERT_model(**encoding)
        esm_BERT=output.pooler_output

        #fusion
        fused_embeddings=torch.cat([esm_ViT,esm_BERT],dim=1)
        logits=self.FusionLayer(fused_embeddings)
        loss=self.loss_fn(logits,labels)
        return loss