# AlphaBetaFusion
A multimodal transformer architecture that combines **Vision Transformers (ViT)** and **BERT-based protein embeddings** for classifying protein molecules using contact maps and generated FASTA sequences

## Model Features
- 13.54M params, 10.24M trainable
- Uses **ViT (Vision Transformer)** for contact map image embeddings.
- Uses **BERT-based protein embeddings** (ESM-2) for sequence information.
- Late Fusion layer combines both embeddings for robust classification.
- Mixed precision training (`torch.amp`) and JIT compiler for faster training on GPU.
- Modular dataset loader supporting train, validation, and test splits and RAM loading to slash disk I/O overhead.
- 'accumulation_steps' can be changed to divide and process batches, to simulate larger batch effects in a small gpu (3 batches and 4 steps => effective batch size is 12)
- You can run the model on my pretrained weights, to test it out.



## Dataset downloading
Follow the steps to generate contact maps and FASTA sequences.
1. Download the CATH domain list from: http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt
2. Open domain_extraction.ps1 in VSCode, edit the no of samples required and run the powershell script. Make sure that the ps1 script exists in the same diretory of the domain list file.
3. Edit the directores, and run contactmap.py. It should take about ~230 minutes to download for 24,000 sequnces


## Model evaluations
-Train accuracy lingered around 85-90% after unfreezing layers 7-11 on vit_tiny and last 2 layers on facebook/esm2_t6_8M_UR50D. 
-Validation accuracy is 76%. A ~10% loss in accuracy is observed. 
-My reasioning is that, 10% difference is more attributable to factors like generalization gap, difficult nature of class seperation, and freezing of lower transformer layers (the model can't learn the underlying features of the structures well. But unfreezing lower layers demands an exponential raise in availability of training data and compute clusters, which is simply not feasible) rather than overfitting.



Model trained on RTX 4060 GPU, i7-13650hx. Fine-tuning time for batch_size=24,epoch=60 arrived roughly at 417 minutes.
<img width="857" height="702" alt="image" src="https://github.com/user-attachments/assets/cc4be937-2de3-489d-a69a-0eace3d3f827" />
