import os.path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os

class UTRDataset_combinedenv(Dataset):
    def __init__(self, species = "human", sample="train", gene="train", dataset_dir="../training_rawdata"):
        train_gene = [i.strip() for i in open(os.path.join(dataset_dir, f"{species}_train_gene.txt"))]
        valid_gene = [i.strip() for i in open(os.path.join(dataset_dir, f"{species}_test_gene.txt"))]
        data = pd.read_parquet(os.path.join(dataset_dir, f"{species}.parquet"))

       
        env_dir_data =  os.path.join(dataset_dir, f"../VAE_project/1024_6species/finetune/{species}/{species}_compressed.csv")
        self.mRNA_vae = pd.read_csv(env_dir_data, index_col=0)
        self.ohe_dict = {i.strip(): index for index, i in
                         enumerate(open(os.path.join(dataset_dir, f"{species}_transcript_ids.txt")))}

        train_sample = [i.strip() for i in open(os.path.join(dataset_dir, f"{species}_train_sample.txt"))]
        valid_sample = [i.strip() for i in open(os.path.join(dataset_dir, f"{species}_test_sample.txt"))]


        if sample == "train":
            data = data[data["sample_RPF_id"].isin(train_sample)]
        else:
            data = data[data["sample_RPF_id"].isin(valid_sample)]
        if gene == "train":
            data = data[data["gene_id"].isin(train_gene)]
        else:
            data = data[data["gene_id"].isin(valid_gene)]

        self.data = data.query("RNA_RPKM > 0.1 ")
        self.data = self.data.reset_index(drop=True)
        folds = 5
        #if species == "yeast":
         #   folds = 3.5
        #elif species == "chicken":
        #    folds = 3
        #elif species == "C.elegans":
         #   folds = 3.5
        
        self.data["RNA_RPKM"] = np.log1p(self.data["RNA_RPKM"] * folds)
        self.data["RPF_RPKM"] = np.log1p(self.data["RPF_RPKM"] * folds)

        self.seq_ohe = np.load(os.path.join(dataset_dir, f"{species}_utr_ohe.npz"))["ohe"]

        self.cds_ohe = np.load(os.path.join(dataset_dir, f"{species}_cds_ohe.npz"))["ohe"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        sample, gene, isoform, RNA, RPF = data
        mRNA_vae = self.mRNA_vae[sample].values
        to_return = {"utr": self.seq_ohe[self.ohe_dict[isoform]],
                     "cds": self.cds_ohe[self.ohe_dict[isoform]],
                     "RNA": RNA,
                     "RPF": RPF,
                     "mRNA_vae": mRNA_vae}
        return to_return

class LabeledDataset(Dataset):
    def __init__(self, dataset, species_id):
        self.dataset = dataset
        self.species_id = species_id  # 0: human, 1: mouse, 2: yeast
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]  # 原始数据和标签
        return data, self.species_id  # 额外返回物种 ID

if __name__ == '__main__':
    dataset = UTRDataset_combinedenv(species = "A.thaliana")

    print("load finished")
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(dataset.data["gene_id"].dtypes)
    for datas in tqdm(data_loader):
        ...
