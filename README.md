# UTR-CODE


<p>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="License">
    </a>
</p>

## 📣 Introduction
___
The 5’ untranslated region (5’ UTR) is a primary driver of protein expression, yet the complexity of its regulatory code has hindered our ability to decode translation regulation, understand variant effects and engineer effective mRNAs. To address this, we developed UTR-CODE, a deep learning model trained on 1,586 samples with paired RNA-seq and Ribo-seq datasets across six species. UTR-CODE demonstrates strong cross-species generalizability and outperforms existing tools across diverse species and platforms.

![img.png](static/pipeline.png)

# ✨  Online website
We provide a [web server](http://www.ai4bio.org/UTR-CODE/) for predicting and optimizing sequences.


## 🔰 Installation

### 1、 Install dependencies
We tested the UTR-CODE on Python 3.10.
```shell
git clone https://github.com/UTR-CODE/UTR-CODE
cd UTR-CODE
pip install -r requirements.txt

```
### 2、Download training raw data
Download and decompress the [training raw data](https://doi.org/10.6084/m9.figshare.30153784.v2) from figshare 


# 📝 Usage

##  Training  model
You can train the UTR-CODE  with the simple example.
```shell
python train.py
```


## ✨ Applications


### 1. Predicted TE from single mRNA sequence
```shell
cd script
python Pred_single.py --weights epochs/best.epoch \
--utr5 GCTACGATCGATCGATCGACTAG \
--utr3 CCACAACCACTGAGT \
--cds CGTACGCTAGCTAGCAT 
```
### 2. Predicted TE from batch mRNA sequence
```shell
cd script
python script/Pred_batch.py --file batch_mRNA.csv --output batch_mRNA.result.csv

```

### 3. Optimize 5'UTR sequence
```shell
python ISM.py --weights epochs/best.epoch \
--utr5 GCTACGATCGATCGATCGACTAG \
--utr3 CCACAACCACTGAGT \
--cds CGTACGCTAGCTAGCAT 
```


### Advanced
If users want train UTR-CODE with custom data, we provide the [tutorial](script/prepare_Data.md) prepare input data.

## 😉 Author
UTR-CODE are maintained by: * [@HeXin](https://github.com/TcbfGroup)


For more Help, Please leave a message in the issue, 
I will reply as soon as possible.




## 📃 License

MIT 