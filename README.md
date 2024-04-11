# Contrastive Learning for Compact Single Image Dehazing

This repository is an unofficial implementation of [Contrastive Learning for Compact Single Image Dehazing](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Contrastive_Learning_for_Compact_Single_Image_Dehazing_CVPR_2021_paper.pdf).

## Requirements

### To install requirements:

```setup
pip install -r requirements.txt
```

### To download datasets:

Download the Synthetic Objective Testing Set (SOTS) [RESIDE]:

https://www.kaggle.com/datasets/balraj98/synthetic-objective-testing-set-sots-reside

Download the Indoor Training Set (ITS) [RESIDE-Standard]:
https://www.kaggle.com/datasets/balraj98/indoor-training-set-its-residestandard

Create dataset directory and extract datasets into dataset folder such that the file structure looks similar to this:

    
    ├── ...
    ├── dataset 
    │   ├── indoor-training-set-its-residestandard
    │   └── synthetic-objective-testing-set-sots-reside
    └── ...

[//]: # (>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)

## Training

To train the model(s) in the paper, run this command:

```train
python train.py
```

[//]: # (>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.)

[//]: # (## Evaluation)

[//]: # ()
[//]: # (To evaluate my model on ImageNet, run:)

[//]: # ()
[//]: # (```eval)

[//]: # (python eval.py --model-file mymodel.pth --benchmark imagenet)

[//]: # (```)

[//]: # ()
[//]: # (>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results &#40;section below&#41;.)

[//]: # (## Pre-trained Models)

[//]: # ()
[//]: # (You can download pretrained models here:)

[//]: # ()
[//]: # (- [My awesome model]&#40;https://drive.google.com/mymodel.pth&#41; trained on ImageNet using parameters x,y,z. )

[//]: # ()
[//]: # (>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained &#40;if applicable&#41;.  Alternatively you can have an additional column in your results table with a link to the models.)

[//]: # (## Results)

[//]: # ()
[//]: # (Our model achieves the following performance on :)

[//]: # ()
[//]: # (### [Image Classification on ImageNet]&#40;https://paperswithcode.com/sota/image-classification-on-imagenet&#41;)

[//]: # ()
[//]: # (| Model name         | Top 1 Accuracy  | Top 5 Accuracy |)

[//]: # (| ------------------ |---------------- | -------------- |)

[//]: # (| My awesome model   |     85%         |      95%       |)

[//]: # ()
[//]: # (>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. )

[//]: # ()
[//]: # (## Contributing)

[//]: # ()
[//]: # (>📋  Pick a licence and describe how to contribute to your code repository. )
