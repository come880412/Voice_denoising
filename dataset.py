from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import torch
import numpy as np
import tqdm
import math
import torch.nn.functional as F

class Denoise_data(Dataset):
    def __init__(self, root, mode, stride=16000, length=64000, pad=True):
        self.root = root
        self.stride = stride
        self.length = length
        self.mode = mode
        self.pad = pad

        self.file = []
        
        if mode == 'train' or mode == 'val':
            data = np.loadtxt(os.path.join(root, mode + '.csv'), delimiter=',', dtype=np.str)
            for noise_data, clean_data, audio_length in data:
                
                noise_data_path = os.path.join(root, 'train', noise_data)
                clean_data_path = os.path.join(root, 'train', clean_data)
                if mode == 'train':
                    if pad:
                        examples = int(math.ceil((int(audio_length) - self.length) / self.stride) + 1)
                    else:
                        examples = (int(audio_length) - self.length) // self.stride + 1
                    for i in range(examples):
                        start_frame = i * self.stride
                        self.file.append([noise_data_path, clean_data_path, start_frame])
                elif mode == 'val':
                    self.file.append([noise_data_path, clean_data_path])
        
        elif mode == 'test':
            data_path = os.path.join(root, 'test')
            data_list = os.listdir(data_path)
            data_list.sort()

            for audio_name in data_list:
                self.file.append([os.path.join(data_path, audio_name), audio_name])

    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            noise_data_path, clean_data_path, start_frame = self.file[index]
            out_noise, _ = torchaudio.load(noise_data_path, frame_offset = start_frame, num_frames = self.length)
            out_clean, _ = torchaudio.load(clean_data_path, frame_offset = start_frame, num_frames = self.length)

            out_noise = F.pad(out_noise, (0, self.length - out_noise.shape[-1]))
            out_clean = F.pad(out_clean, (0, self.length - out_clean.shape[-1]))

            return torch.FloatTensor(out_noise), torch.FloatTensor(out_clean)
        elif self.mode == 'val':
            noise_data_path, clean_data_path = self.file[index]
            out_noise, _ = torchaudio.load(noise_data_path)
            out_clean, _ = torchaudio.load(clean_data_path)

            return torch.FloatTensor(out_noise), torch.FloatTensor(out_clean)
        elif self.mode == 'test':
            noise_data_path, audio_name = self.file[index]
            out_noise, _ = torchaudio.load(noise_data_path)

            return torch.FloatTensor(out_noise), audio_name


if __name__ == '__main__':
    data = Denoise_data(root = '../dataset', mode='train', stride=16000, length=64000, pad=True)
    dataloaer = DataLoader(data, batch_size=2, shuffle=True, num_workers=0)
    print(len(data))
    max_length = 0


    for noise, clean in tqdm.tqdm(dataloaer):
        print(noise.shape, clean.shape)