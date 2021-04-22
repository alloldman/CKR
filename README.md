# CKR
## environments:

pytorch==1.3

pytorch-transformers==1.2.0

nltk=3.5

pandas=0.21.0

networkX

## data:
Download data from `https://drive.google.com/drive/folders/1lU6k8DNXThdWXOafHoXC-3UjwCArT84h?usp=sharing`. There are four files: data.zip, cache.zip, img_features.zip and best-ckpt.zip. 
Unzip these zipfile, and put them under the order below.
```
CKR
├──data
├──KB
│  └──cache
├──experiments
│  └──best-ckpt
└──img_features
   └──ResNet-152-imagenet.tsv 
```

## to inference:
`bash run.sh search experiments/best-ckpt/follower_pm_sample2step_imagenet_mean_pooled_1heads_train_iter_9300val_seen_sr_0.547_val_unseen_sr_0.138_ 0`


## to train:
`bash run.sh train 0

