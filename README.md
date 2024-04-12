# DesignQA Benchmark

**DesignQA: A Multimodal Benchmark for Evaluating Large Language Models' Understanding of Engineering Documentation**

Check out the pre-print [here](https://arxiv.org/abs/2404.07917)!

Check out our official website and leaderboard [here](https://design-qa.github.io/)!

## Overview

DesignQA is a novel benchmark aimed at evaluating the proficiency of multimodal large language models (MLLMs) in comprehending and applying engineering requirements in technical documentation. The benchmark is developed in conjunction with the MIT Motorsport team, so that the question-answer pairs are based on real world data and the ~200 page FSAE competition rules. Some key features of the DesignQA benchmark include:

* 1451 question-answer pairs pertaining to design of an FSAE vehicle according to the FSAE competition rules

* 3 benchmark segments - rule extraction, rule comprehension, and rule compliance - that enable fine-grained investigation into a model's strengths and weaknesses when it comes to design according to 

* Automatic evaluation metrics for quick scoring of new MLLMs

* A unique benchmark that requires models to analyze and integrate information from both visual and long-text inputs

Here is a visual overview of the benchmark:

![Dataset Overview](docs/images/designqa_overview.png)

## Dataset
The dataset can be found in the ```dataset``` directory. The dataset is split into three sections: Rule Extraction, Rule Comprehension, and Rule Evaluation. Each dataset section has two different subsets, each of which corresponds with a specific task needed to design according to technical documentation. Examples of the segments and subsets are as follows:

### Rule Extraction
This segment of the dataset tests a model's ability to extract requested information from a lengthy technical document. The questions in this segment do not involve images. The Rule Extraction question-answer pairs are further divided into two subsets: Retrieval QAs and Compilation QAs.

#### Retrieval QAs
These questions ask the model to extract a specific rule from the 2024 FSAE competition rules. The retrieval QAs can be found in ```dataset/rule_extraction/rule_retrieval_qa.csv```. All of the questions have the format:

```
We are a student engineering team designing a vehicle for the FSAE competition. Attached is the FSAE rules document. What does rule {rule_number} state exactly? Answer with only the text of the rule and no other words.
```

An example ground-truth answer (for rule V.1) is:

```
CONFIGURATION The vehicle must be open wheeled and open cockpit (a formula style body) with four wheels that are not in a straight line.
```

#### Compilation QAs
These questions ask the model to find all rules in the FSAE competition rules relevant to a specific term. The compilation QAs can be found in ```dataset/rule_extraction/rule_compilation_qa.csv```. All of the questions have the format:

```
We are a student engineering team designing a vehicle for the FSAE competition. Attached is the FSAE rules document. Please list all rules relevant to {term}. Answer with only the rule numbers (i.e.: AA.1.1.1) separated by commas and no other words. The rules relevant to `Aerodynamic/Aerodynamics` are:
```

An example ground-truth answer (for rule the term `Aerodynamic/Aerodynamics`) is:

```
['T.7', 'T.7.1', 'T.7.1.1', 'T.7.1.3', 'T.7.2.1', 'T.7.2.2', 'T.7.3.1', 'T.7.3.3', 'T.7.4', 'T.7.5', 'T.7.6', 'T.7.6.3', 'T.7.7.1', 'IN.8.2', 'IN.8.2.1', 'IN.8.2.3', 'T.7.1.2', 'T.7.1.4', 'T.7.1.5', 'T.7.2', 'T.7.2.3', 'T.7.2.4', 'T.7.3', 'T.7.3.2', 'T.7.6.1', 'T.7.6.2', 'T.7.7', 'T.7.7.2', 'IN.8.2.2', 'GR.6.4.1', 'V.1.1', 'V.1.4.1']
```

### Rule Comprehension
This semgnet of the dataset tests a model's ability to understand the terms and definitions presented within a specific rule or requirement. The questions in this segment involve images. The Rule Comprehension question-answer pairs are further divded into two subsets: Definition QAs and Presence QAs.

#### Definition QAs


#### Presence QAs

## Automatic Evaluation Metrics


## Evaluating Your Model
Implementation of existing MLLM model evaluation can be found in the ```eval``` directory.

## Citations
TODO