import torch
import os
from config import DATA_DIR
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DatasetLoader:
    def __init__(self,root_dir=DATA_DIR):
        self.transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)
        ])

        self.contact_maps=datasets.ImageFolder(root_dir)
        self.samples=self.contact_maps.samples  

        total_size=len(self.samples)
        generator=torch.Generator().manual_seed(42)
        indices=torch.randperm(total_size,generator=generator).tolist()

        self.train_size=int(0.7*total_size)
        self.val_size=int(0.15*total_size)
        self.test_size=total_size-self.train_size-self.val_size

        self.train_idx=indices[:self.train_size]
        self.val_idx=indices[self.train_size:self.train_size+self.val_size]
        self.test_idx=indices[self.train_size+self.val_size:]

    def _load_sample(self,img_path,label):
        img=Image.open(img_path).convert("RGB")
        img_tensor=self.transform(img)
        fasta_path=img_path.replace(".png",".fasta")
        if os.path.exists(fasta_path):
            with open(fasta_path,"r") as f:
                lines=f.readlines()
            sequence="".join([line.strip() for line in lines if not line.startswith(">")])
        else:
            sequence=""
        return img_tensor,label,sequence

    def _create_loader(self,indices,batch_size,shuffle=True):
        subset=[self.samples[i] for i in indices]
        def collate_fn(batch):
            loaded=[self._load_sample(img_path, label) for img_path, label in batch]
            images,labels,sequences=zip(*loaded)
            images=torch.stack(images)
            labels=torch.tensor(labels)
            return images,labels,sequences
        return DataLoader(subset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn)

    def TrainSplit(self,batch_size):
        return self._create_loader(self.train_idx,batch_size,shuffle=True)
    def ValSplit(self,batch_size):
        return self._create_loader(self.val_idx,batch_size,shuffle=False)
    def TestSplit(self,batch_size):
        return self._create_loader(self.test_idx,batch_size,shuffle=False)
