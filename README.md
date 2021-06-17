# Predicting Prognosis in COVID-19 Patients using Machine Learning and Readily Available Clinical Data

This repository is the official implementation of [Predicting Prognosis in COVID-19 Patients using Machine Learning and Readily Available Clinical Data](https://www.medrxiv.org/content/10.1101/2021.01.29.21250762v1). 

## Requirements

After cloning the repository, make sure all directories and subdirectories contained are in the Matlab path.

## Training, Evaluation, and Results

dxCortex_example.m demonstrates how to set up, train, and evaluate a basic dxCortex model without trees like those used in the child classifiers in the paper.  dxCortexForest_example.m demonstrates the same but for the dxCortex models with tree like those used in the top classifiers in the hierarchies from the paper.  The results produced by the script are out-of-bag performance estimates on the synthetic development set and validation performance on the synthetic validation set.  


## Contributing

Please see License.txt for more information, and feel free to contact the authors with any questions!
