# Voice_denoising
Competition URL: https://aidea-web.tw/topic/8d381596-ee9d-45d5-b779-188909ccb0c8 (Private 6th place)

# Method
We refer to the method of [1] to solve this problem. If you want to know more details, I recommend you see the original paper of Demucs [2]. The model is shown as follow:
<p align="center">
<img src="https://github.com/come880412/Voice_denoising/blob/main/image/demucs.png" width=50% height=50%>
</p>

# Evaluation metric
The PESQ value is used to evaluate the quality of the denoised audio. The formulation of PESQ value can be found in https://github.com/ludlows/python-pesq

# Getting Started
- Clone this repo to your local
``` bash
git https://github.com/come880412/Voice_denoising
cd Voice_denoising
```

### Data preprocessing
First, you need to download the dataset from [here](https://aidea-web.tw/topic/8d381596-ee9d-45d5-b779-188909ccb0c8). Then, put your data to the directory `../dataset/`. After that, you can use the following command to split the training/validation sets:
``` bash
python preprocessing.py
```

### Inference
You can use the following command to test this result:
```bash
$ python test.py
```
- The denoised audio will be generated on the directory `./output` automatically.

### Training
If you want to train the model from scratch, you can use the following command:
```bash
$ python train.py
```

- If you have any implementation problems, please feel free to e-mail me! come880412@gmail.com

# Reference
[1] https://github.com/facebookresearch/denoiser
[2] Defossez, A., Synnaeve, G., & Adi, Y. (2020). Real time speech enhancement in the waveform domain. arXiv preprint arXiv:2006.12847.
