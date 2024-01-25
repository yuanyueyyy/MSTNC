## PyTorch Demo of the Hyperspectral Image Classification method - MSTNC.
Using the code should cite the following paper:
Bai, Yu and Liu, Dongmin and Wu, Haoqi and Zhang, Lili, "MSTNC: Multi-Strategy Triple Network Classifier for Small-Sample Hyperspectral Image Classification" in IEEE Transactions on Geoscience and Remote Sensing

@ARTICLE{
    author={Bai, Yu and Liu, Dongmin and Wu, Haoqi and Zhang, Lili},  
    journal={IEEE Transactions on Geoscience and Remote Sensing},   
    title={MSTNC: Multi-Strategy Triple Network Classifier for Small-Sample Hyperspectral Image Classification},   
    year={2024}, 
    volume={},  
    number={},  
    pages={1-17},  
    doi={}
}



# Description.
 In recent years, there has been a flourishing development of deep learning (DL) methods in the field of hyperspectral image (HSI) classification. However, most of the existing HSI classification methods rely heavily on a large number of labeled samples, resulting in very expensive classification costs. How to use small-sample data to efficiently classify has become a critical issue. In this paper, a multi-strategy triplet network classifier (MSTNC) learning network is proposed to solve the limited labeled samples issue for HSI classification. First, we designed a triplet network classifier (TNC) with low sample dependency as the backbone for model training. The TNC, which mainly consists of an encoder module and a projection head, as well as a classifier module, is used to extract HSI image features and perform classification. Secondly, for classification tasks with multiple classes and limited samples, a feature mixture-based active learning (FMAL) method is used to query the unlabeled samples with the strongest uncertainty. These queried samples are labeled and added to the training set, and the TNC uses the new training set to further extract deep semantic features. Finally, the Pseudo-Active Learning (PAL) strategy is proposed innovatively. The predicted values of the classifier are used to label the samples of the candidate set, so as to expand the training set without increasing the labeling cost and improve the classification accuracy of the model. Extensive experiments are conducted on three benchmark HSI datasets, and MSTNC outperforms several state-of-the-art methods. For study replication, the code developed for this study is available at https://github.com/yuanyueyyy/MSTNC.git

## Requirements
```
* pytorch
* time
* argparse
* os
* numpy
* random
* sklearn
* metrics
```

