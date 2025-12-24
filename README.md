# RB-SCD: Semantic Change Detection of Roads and Bridges
This repository contains the official implementation and dataset samples for the paper:
"Semantic Change Detection of Roads and Bridges: A Fine-grained Dataset and Multimodal Frequency-driven Detector"  




## Introduction
Accurate detection of road and bridge changes is crucial for urban planning but challenging due to structural continuity requirements and semantic ambiguities. We propose:
RB-SCD Dataset: The first fine-grained semantic change detection dataset specifically for traffic infrastructure, featuring 11 distinct change transition categories.
MFDCD: A Multimodal Frequency-Driven Change Detector that leverages:
Dynamic Frequency Coupler (DFC): Models linear structural continuity via wavelet transforms.
Textual Frequency Filter (TFF): Resolves semantic ambiguity using text-guided frequency filtering.  

## Dataset Preview (RB-SCD)
We provide a preview of the RB-SCD dataset in the dataset_samples/ folder.  


## Model Architecture (MFDCD)
We provide the core implementation of our proposed modules in the models/ directory:
models/DFC.py: Implementation of the Dynamic Frequency Coupler.
models/TFF.py: Implementation of the Textual Frequency Filter.


