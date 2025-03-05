# RS-NormGAN
The official code of RS-NormGAN<br>
Paper: J. Miao, S. Li, X. Bai, W. Gan, J. Wu and X. Li, "RS-NormGAN: Enhancing change detection of multi-temporal optical remote sensing images through effective radiometric normalization," ISPRS Journal of Photogrammetry and Remote Sensing, vol. 221, pp. 324-346, 2025. https://doi.org/10.1016/j.isprsjprs.2025.02.005.<br>
We proposed a pseudo invariant feature (PIF) - inspired weakly supervised generative adversarial network (GAN) for remote sensing images radiometric normalization, which is beneficial for applications.

![overall](https://github.com/user-attachments/assets/e90d37b3-ada9-40ba-96fa-919e57d2ed02)

# Environment
The experments were conducted on Dell Workstation with Nvidia Quadro RTX5000 GPU and Ubuntu18.04.<br>
Crucial packages:<br>
Python 3.7.11<br>
cudatoolkit 11.3.1<br>
PyTorch 1.10.2 GPU version + torchvision 0.11.3 + torchaudio 0.10.2 <br>
numpy 1.21.5 <br>
pillow 9.0.1 <br>
# Normalization
## Data Acquisition
https://pan.baidu.com/s/1P24MCokRd9Icbz8drBFARA?pwd=mtRS <br>
Put the folder "datasets" inside the folder "Normalization".<br>
## Train
```python train.py```<br>
The configs can be modified in the "options/base_options.py and options/train_options.py" <br>
## Test
```python train.py --dataroot Your_path_to_data --model testdoubleG --no_dropout --use_wandb --model_suffix _A --load_size 256 --name Your_experiment_name --ngf 64 --netG ResUnet```<br>
The configs can be modified in the " options/test_options.py" <br>

# Change Detection
## DataSet
  **file path**
- Datasets
  - Experiment_Name
    - train
      - A
        - Normalization
      - B
      - label
    - test
    - val
## Train
for Sentinel Hefei Change Detection dataset
```python train_SHCD.py```<br>
for Google Earth Seasonal Difference dataset
```python train_GESD.py```<br>
## Test
Modify the loss calculation and CD Network by changing test.py
# Contact
jianhao_miao@whu.edu.cn
