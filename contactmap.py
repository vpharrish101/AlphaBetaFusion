import os
import numpy as np
import pandas as pd
import threading

from io import StringIO
from PIL import Image
from Bio.PDB import PDBParser, PDBList
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from Bio.PDB.Polypeptide import three_to_one as _bp_three_to_one
except Exception:
    _bp_three_to_one = None

def three_to_one(resname):
    if _bp_three_to_one:
        try:
            return _bp_three_to_one(resname)
        except Exception:
            pass
    mapping={
        'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
        'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
        'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'
    }
    return mapping.get(resname.strip().upper(),'X')


classes={
    "alpha"     : r'D:\Python311\ViT-BERT\CATH\contact_maps\alpha',
    "beta"      : r'D:\Python311\ViT-BERT\CATH\contact_maps\beta',
    "alpha_beta": r'D:\Python311\ViT-BERT\CATH\contact_maps\alpha_beta',
    "fewss"     : r'D:\Python311\ViT-BERT\CATH\contact_maps\fewss'
}

for path in classes.values():
    os.makedirs(path,exist_ok=True)

csv_paths={
    "alpha"     : r'D:\Python311\ViT-BERT\CATH\contact_maps\alpha\alpha_sample.csv',
    "beta"      : r'D:\Python311\ViT-BERT\CATH\contact_maps\beta\beta_sample.csv',
    "alpha_beta": r'D:\Python311\ViT-BERT\CATH\contact_maps\alpha_beta\alpha_beta_sample.csv',
    "fewss"     : r'D:\Python311\ViT-BERT\CATH\contact_maps\fewss\fewss_sample.csv'
}

pdb_files={k:pd.read_csv(v,header=None).iloc[:,0].str[:4] for k,v in csv_paths.items()}

distance_threshold=8.0
min_contacts=50
max_samples=3
sample_counters={k: 0 for k in pdb_files.keys()}
counter_lock=threading.Lock()
pdbl=PDBList()

def compute_contact_map(coords,threshold=8.0):
    arr=np.array(coords)
    diff=arr[:,None,:]-arr[None,:,:]
    dists=np.sqrt(np.sum(diff**2,axis=-1))
    return (dists<=threshold).astype(np.float32)


def process_pdb(pdb_id,class_key):
    try:
        pdb_path=pdbl.retrieve_pdb_file(pdb_id,pdir="pdb_temp",file_format="pdb")
        if not os.path.exists(pdb_path):
            print(f"{pdb_id} download failed")
            return False

        with open(pdb_path,'r') as f:
            pdb_str=f.read()

        structure=PDBParser(QUIET=True).get_structure(pdb_id,StringIO(pdb_str))
        coords_list=[]
        sequences={}

        for model in structure:
            for chain in model:
                ca_atoms=[res['CA'].coord for res in chain if 'CA' in res]
                seq_chars=[]
                for res in chain:
                    if res.get_id()[0] != ' ':
                        continue
                    aa=three_to_one(res.get_resname())
                    seq_chars.append(aa)
                if seq_chars:
                    sequences[chain.id]=' '.join(seq_chars)
                if len(ca_atoms)>=2:
                    coords_list.extend(ca_atoms)

        if len(coords_list)<2 or not sequences:
            return False

        cm=compute_contact_map(coords_list,distance_threshold)
        if cm.sum()<min_contacts:
            return False

        cm_rgb=np.stack([cm,cm,cm],axis=-1)*255
        img=Image.fromarray(cm_rgb.astype(np.uint8)).resize((224,224))
        out_path=os.path.join(classes[class_key],f"{pdb_id}.png")
        img.save(out_path)

        fasta_path=os.path.join(classes[class_key],f"{pdb_id}.fasta")
        with open(fasta_path,'w') as fh:
            for chain_id,seq in sequences.items():
                fh.write(f">{pdb_id}_{chain_id}\n")
                for i in range(0, len(seq),80):
                    fh.write(seq[i:i+80]+"\n")

        with counter_lock:
            if sample_counters[class_key]<max_samples:
                sample_counters[class_key]+=1
                return True
            else:
                return False

    except Exception as e:
        print(f"{pdb_id} failed: {e}")
        return False

def batch_process_pdbs(pdb_dict,max_workers=8):
    futures={}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for class_key,ids in pdb_dict.items():
            for pdb_id in ids:
                if sample_counters[class_key]>=max_samples:
                    break
                f=executor.submit(process_pdb,pdb_id,class_key)
                futures[f]=class_key

        for f in as_completed(futures):
            class_key=futures[f]
            f.result()

if __name__ == "__main__":    
    batch_process_pdbs(pdb_files)
    print("done.")
