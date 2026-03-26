# macbert-chinese-mine-accident-causation-classification

This repository provides the source code for Chinese mine accident causation text classification based on MacBERT-BiLSTM. It is mainly used for the automatic identification and classification of accident causation texts and supports the reproduction of the classification experiments reported in the paper.

## Project Description
This project constructs a Chinese text classification model for mine accident causation factors based on MacBERT-BiLSTM. The released source code covers model training, evaluation, repeated random split experiments, tokenizer configuration, and related parameter settings, which can provide a reference for related research.

## Main Contents
- `train_macbert.py`: main training and evaluation script for the MacBERT-BiLSTM classification model
- `random_split_experiments.py`: repeated random split experiments for model stability verification
- `config.json`: model configuration file
- `tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`, `added_tokens.json`, `vocab.txt`: tokenizer-related files

## Model Settings
The core model is built on MacBERT and BiLSTM for six-class text classification. The main experimental settings include:
- train/test split ratio: 80% / 20%
- max sequence length: 16
- learning rate: 1e-5
- batch size: 16
- number of epochs: 5
- weight decay: 0.01

## Environment
Recommended environment:
- Python 3.10+
- PyTorch
- transformers
- datasets
- pandas
- numpy
- scikit-learn
- matplotlib
- openpyxl

## Usage
1. Prepare the dataset file in `.xlsx` format.
2. Modify the local data path and model path in the script if necessary.
3. Run `train_macbert.py` to train and evaluate the model.
4. Run `random_split_experiments.py` to reproduce repeated random split experiments.

## Notes
This repository releases the core source code and essential configuration files for the MacBERT-BiLSTM classification task. Large local cache files, checkpoints, and other non-essential runtime files are not included.

## Citation
If you use this code in your research, please cite the corresponding paper.
