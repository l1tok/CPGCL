# CPGCL: Cluster-Perceptive Graph Contrastive Learning for Community Detection

Code implementation of paper "Cluster-Perceptive Graph Contrastive Learning for Community Detection".
Accepted by ICASSP202.

## Requirements
Install the dependencies: `pip install -r requirements.txt`.
- munkres==1.1.4
- numpy==2.2.1
- scikit_learn==1.6.0
- scipy==1.14.1
- torch==2.4.0

## Usage

First, unzip the dataset, which is saved in `.txt` file format:

`unzip data.zip`

Then run `main.py` to train the model (taking AMAP as an example):

`python main.py --dataset amap --epoch 500 --lr 7e-4 --beta 0.4`
