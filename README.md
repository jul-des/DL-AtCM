## Diagnosing atrial cardiomyopathy from 12 lead ECG with deep learning
This github contains the code to the paper: Predicting cardiac magnetic resonance imaging markers of atrial cardiopathy from 12 lead ECG using deep learning. A UK Biobank study. 

Citation:
```
"Diagnosing atrial cardiomyopathy from 12 lead ECG with deep learning. A UK Biobank study."
Julian Deseoe, Ezequiel de la Rosa, Martin Haensel, Lisa Herzog, Neda Davoudi, Andreas Luft, 
Beate Sick, Jan Steffel, Alexander Breitenstein, Gregory Lip, Bjoern Menze, Susanne Wegener."
```

In this paper we finetune a ECG foundational model (ECG-FM) published by McKeen et al. to predict cardiac imaging markers of atrial cardiopathy. We call this model AC_CMR. We then show that the predicted imaging markers allow the prediction of atrial fibrillation, outperforming previous approaches. Details concerning ECG-FM can be found under:

```
"McKeen K, Masood S, Toma A, Rubin B, Wang B. ECG-FM: an open electrocardiogram foundation model. Jamia Open. 
1. Oktober 2025;8(5):ooaf122."
```
Be sure to also checkout the corresponding github repository under: https://github.com/bowang-lab/ecg-fm

The pretrained ECG-FM model can be downloaded under: https://huggingface.co/wanglab/ecg-fm/tree/main. We finetune the mimic_iv_ecg_physionet_pretrained.pt model.

We compare against two baselines. One baseline is created by extracting P wave indices (PWI) from the ECGs. The PWI can be extracted using the Jupyter Notebook in the folder P wave indices.

The second baseline is created by finetuning ECG-FM to identify patients with previous diagnosis of atrial fibrillation directly. We call this model DL_AF

## Installation

First clone this repository. We recommend creating a conda environment using the provided environment.yml file
ECG-FM was built in collaboration with fairseq-signals, relevant dependencies are all installed via the environment.yml file

````console
git clone https://github.com/jul-des/DL-AtCM
cd DL-AtCM
conda env create -f environment.yml -n dl_atcm
conda activate dl_atcm
````

Next, download the ECG-FM pretrained model from the link shown above. We finetuned the mimic_iv_ecg_physionet_pretrained.pt model.

Our finetuned models can be downloaded from "XXXXX"

## Preprocessing

We preprocess and segment the ECGs according to the specifications of McKeen et al. Their pipeline contains resampling to 500 Hz, normalization of ECGs and segmentation into 5 second segments. ECG-FM was built in collaboration with fairseq-signals, where the preprocessing pipeline is implemented. Details can be found under https://github.com/Jwoo5/fairseq-signals/tree/master/scripts/preprocess/ecg. For preprocessing of the CODE15% Dataset a solution is already implemented. For other datasets the example_records.py and example_signals.py scripts will have to be adapted according to the specifications.

Our deep learning models expect ECGs of shape  (N, 12, 2500) as input i.e. 12-channel, 5 second ECGs resampled to 500 Hz. The leads should be in the order: 'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'. Missing leads can be masked with zeroes. For PWI extraction any ECG length can be handled, although results become more reliable the more P waves are present in the ECG signal.

## Model Finetuning

We have built two different models, both by finetuning ECG-FM. We finetune one model (AC_CMR) to predict cardiac magnetic resonance imaging features of atrial cardiopathy from 12 lead ECG. We finetune a second model (DL_AF) to identify patients with previous diagnosis of atrial fibrillation directly.

The models and helper functions are defined in ModelClassAtCM_CMR.py and ModelClassDL_AF respectively. The hyperparameters are defined in config.yaml and the finetuning is performed in model_finetuning_AtCM_cmr.py and model_finetuning_dl_af.py respectively. 

To run the scripts for training, the paths at the top of the files have to be defined.  

## Model Evaluation

In the directory Model Evaluation, the provided Jupyter Notebooks can be run to get model predictions.

## Getting P wave indices

The Jupyter Notebook in the directory P wave indiced can be run to get P wave indices (P wave durtation, PQ intervall, P Axis and P amplitude in lead II) of ECGs. We ran the code on the preprocessed but not segmented 12 lead ECGs, returned by the described preprocessing pipeline.

## Saliency analysis 
The Jupyter Notebook in the directory Saliency can be run to get saliency maps for 12 lead ECGs

## Questions
Inquiries can be directed to julian.deseoe@uzh.ch
