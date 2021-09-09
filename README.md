# Lexico-acoustic models with attention in Dialog Act Classification
Esra Dönmez, Christoph Schaller, Wei Zhou

Project report: [Here] (https://github.com/esradonmez/Dialogue_act_classification/blob/main/report.pdf)

## Abstract
Dialog act classification (DAC) is of significant importance in spoken dialog systems. Recent works have proposed several successful neural models for dialog act classification, many of which only explored the task by taking advantage of the transcripts of the audio files and building lexical models. In 2018, Ortega and Vu. proposed a lexico-acoustic neural-based model for this task to utilize the acoustic information in combination with the lexical features. In their experiments, they use convolutional neural networks (CNNs) \cite{kalchbrenner} to learn both the lexical and the acoustic features. Moreover, pretrained language models have been successfully used in recent years in various tasks to learn contextual embeddings from natural language input. In this paper, we experiment with both lexical and acoustic models in DAC. We implemented two models, one for the textual input and one for the acoustic input, and different ways to combine these modalities for multimodal learning in DAC. We evaluate our models on a subset of the SwDA (Switchboard Corpus) (Calhoun et al., 2010) and compare the results of the two models as well as the combined model. Furthermore, we conduct both quantitative and qualitative error analysis to investigate the strengths and weaknesses of these models. We also run an ablation study to examine the contributions of the individual components in our lexical model. Our results show that lexical model is sufficient to learn the features that are represented by both modalities, i.e. the acoustic model might not learn additional useful information that is not present in lexical features when combined with a powerful lexical model.

## Requirements

All requirements are listed in [pyproject.toml](pyproject.toml)  
Install them using pip or [poetry](https://python-poetry.org/)  
  
Make sure the required data is available.

## Preprocessing 
To extract MFFCs from the audio files, run
`python preprocess.py` from the root folder

## training
- To train the models, run the following commands from the root folder.
  - For combined model:
```
python train_combined.py
```
  - For lexical model:
```
python train_lexical.py
```
- For acoustic model:
```
python train_acoustic.py
```
