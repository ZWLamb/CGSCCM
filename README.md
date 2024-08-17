#CGSCCM for Nuclei Instance segmentation
Authorized deployment of CGSCCM for nuclei instance segmentation in histological images.
### Setup Enviorment
For our experiments, we utilized the Anaconda package manager and included an `environment.yml` to replicate our setup. To establish the environment, please execute the following command:
```console
conda create --name myenv
conda activate myenv
conda install --file environment.txt
```
### Datasets
We have trained the model on three Datasets that are CoNSeP, PanNuke and Lizard. Below are the links to download these datasets:
- [CoNSeP](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/)
- [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)
- [Lizard](https://warwick.ac.uk/fac/cross_fac/tia/data/lizard)

### Training
Before beginning model training, review config.py to configure the essential hyperparameters. To start the training, execute the command below:
```console
python train.py
```
### Model Evaluation
For evaluation, we are utilizing the following metrics: `Dice Score`, `Aggregated Jaccard Index (AJI)`, `Panoptic Quality (SQ)`, `Detection Quality (DQ)`, and `Panoptic Quality (PQ)`. To perform the model evaluation, use the command provided below:
```console
python verify.py
```