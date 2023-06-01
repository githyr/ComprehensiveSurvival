# ComprehensiveSurvival
This is an implementation of Cancer survival prediction by learning comprehensive deep feature representation for multiple types of genetic data in Pytorch.
## Requirements
  * Python=3.6.5  
  * Pytorch=1.6.0  
  * Torchvision=0.7.0
## Datasets
The model is trained on glioblastoma multiforme (GBM), kidney renal clear cell carcinoma (KRCCC), lung squamous cell carcinoma (LSCC) and breast invasive carcinoma (BIC) dataset, where each dataset are splited into two parts: 70% samples for training, 20% samples for validation, and the rest samples for testing.  We utilize the classification accuracy, Recall, Precision to evaluate the performance of all the methods.
## Implementation

#Train/Test the model on GBM dataset

`` python siamese_shared_specific.py ``
