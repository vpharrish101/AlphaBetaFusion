# AlphaBetaFusion
A multimodal transformer architecture that combines **Vision Transformers (ViT)** and **BERT-based protein embeddings** for classifying protein molecules using contact maps and generated FASTA sequences

## Features
- Uses **ViT (Vision Transformer)** for contact map image embeddings.
- Uses **BERT-based protein embeddings** (ESM-2) for sequence information.
- Fusion layer combines both embeddings for robust classification.
- Mixed precision training (`torch.amp`) for faster training on GPU.
- Modular dataset loader supporting train, validation, and test splits.

## Dataset downloading
Follow the steps to generate contact maps and FASTA sequences.
1. Download the CATH domain list from: http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt
2. Open domain_extraction.ps1 in VSCode, edit the no of samples required and run the powershell script. Make sure that the ps1 script exists in the same diretory of the domain list file.
3. Edit the directores, and run contactmap.py. It should take about ~230 minutes to download for 24,000 sequnces


I have yet to test the model, so minor bugs are expected. Full README will be pushed along with updated weights, after testing. 
