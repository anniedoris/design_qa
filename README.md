# design_qa

This repo is a reference-dependent document-grounded VQA/long-text dataset pertaining to the FSAE rules. We provide the dataset and methods to evaluate select MLLMs.

## Dataset
The dataset can be found in the ```dataset``` directory. The dataset is split into three sections: Rule Extraction, Rule Comprehension, and Rule Evaluation. Each dataset section has several different task datasets, which are QAs that can be fed directly to a model for model evaluation. All of these task datasets can be found in the ```dataset``` directory (under their relevant section) and end with ```_qa.csv```. Scripts for dataset generation can be found in the ```scripts``` directory.

## Evaluation
Implementation of existing MLLM model evaluation can be found in the ```eval``` directory.