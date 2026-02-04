# Plankton segmentation: A transfer learning and active learning approach

Working on the [ZooScanNet](https://www.seanoe.org/data/00446/55741/) planktons dataset, I needed to get masks separation planktons from noise and background. 

Difficulties:
- Traditional image processing techniques were not easy to implement because of noise in the background
- Using pre-trained models for segmentations like SAM was not successful as these models are usually trained on datasets such as ImageNet and that the domain shift is too important
- From-scratch manual annotation is possible but time consuming (84 classes in the dataset)

To answer the problem I made my own approach reducing the difficulties linked to the nature of the dataset:

### Transfer Learning

To reduce the difficulty of manual image annotations I trained a model on a dataset with a cleaner background ([Pelgas Dataset](https://www.seanoe.org/data/00829/94052/)).

Working on a cleaner dataset allowed to extract some masks using a traditionnal image processing method. 

The idea is to hope that a model trained on this dataset can perform good enough on the ZooScanDataset and allow me get a starting point for manual annotation which will save me a lot of time. 

#### Phase 1: Annotation of a validation dataset

For the next steps, I needed to annotate a validation set from the ZooScanNet dataset so that I could measure how good my future models perform. 

A custom annotation tool was built using Napari to manually annotation around 70 images from the goal dataset. 

#### Phase 2: Find an architecture and hyperparameters suitable for generalization

The first step in my approach corresponds to finding an architecture and hyperparamters which will allow a model trained on PELGAS to be good enough for inference on the ZooScanNet dataset. It also will be a good baseline. 

Since training is fast, I experimented on two architectures and 12 combinations of hyperparameters in a gridsearch:

- Model 1: backbone: resnet34 + UNet
- Model 2: backbone: resnet34 + DeepLabV3Plus

Among the hyperparaters, I included the decision to freeze or unfreeze the backbone. 

Unsurprisingly, simpler and unfrozen models performed better on my test set. This is due to:
- The backbone being pretrained on ImageNet, and thus being able to extract features meaningful for a different yet wider domain of images;
- More complex models overfitting faster on the train set

Overall, all models tend to overfit after a few epochs which is also not surprising. 

(INCLUDE TRAINING CURVES (train loss vs. valid loss over epochs))


This experiment allowed to validate my intuition and to choose the best model for transfer learning:

(INCLUDE BEST MODEL AND PERFORMANCE)

However, the segmentation on ZooScanNet unlabelled images with this unaugmented training is really bad :

<p align="center">
  <img src="./assets/gridsearch_example1.png" width="600">
  <img src="./assets/gridsearch_example2.png" width="600">
  <img src="./assets/gridsearch_example3.png" width="600">
</p>

Oversegmentation and noise are included, making this approach worse than a simple image processing pipeline.

#### Phase 3: Find a set of augmentations to improve generalization

The previous training were done with only very basic transformations. It is highly predicatible that heavier augmentations will result in better performances on the ZooScanNet dataset. To test this hypothesis and find the best set of augmentations, I followed advices from the [Albumentations documentation](https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/) and made a set of 5 augmentations compositions.

These compositions are:

- 'basic': Resize + HorizontalFlip + VerticalFlip + Rotation
- 'occlusion': All of the above + CoarseDropout
- 'affine': All of the above + replacing Rotation by Affine augmentations
- 'domain' All of the above +  GridDistortion + RandomGamma + ISONoise + Blurring + Downscale
- 'specialized': All of the above + FDA/HistogramMatching

((INCLUDE RESULTS FOR THESE TRAINING))

### Active Learning

At this point, I had a decently working model but there were still a lot of inconsistancies when infering on the ZooScanNet dataset. These inconsistancies were not acceptable for the intended application. Since annotation was expensive (I didn't have a lot of time to spend on it), I opted for an active learning approach.

Challenge: Lack of diversity in the sampling of images to label: Prototyping this active learning phase showed me that the basic uncertainty sampling stategy lacked a lot of diversity (mostly showing eggs)

To solve the problem, I made a method inspired by the paper [Active learning for medical image segmentation with stochastic batches](https://arxiv.org/abs/2301.07670). The idea is:

- Instead of computing the uncertainty score on images individually, compute it on batches made from a random sampling in the available unlabelled images
- Use TTA to estimate the uncertainty better


