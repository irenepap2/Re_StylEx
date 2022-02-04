# Reproducibility study for "Explaining in Style: Training a GAN to explain a classifier in StyleSpace"

This repository contains our experiments in reproducing Lang et.al's ["Explaining in Style: Training a GAN to explain a classifier in StyleSpace"](https://arxiv.org/abs/2104.13369).

## Requirements

To install requirements you need to activate the following environment in anaconda:

```setup
conda env create -f environment.yml
```

## Training

To train our implementation of the StylEx model in the paper, run the following notebook in Colab:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

To reproduce the results shown in our paper, run the following notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)


>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

