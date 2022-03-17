import numpy as np
import os
import torchaudio
from tqdm import tqdm

if __name__ == '__main__':
    data_path = '../dataset/train'
    save_path = '../dataset'
    train_val_ratio = 0.9

    data = []
    category_dict = {}
    category_list = []
    data_list = os.listdir(data_path)
    for data_name in tqdm(data_list):
        data_name_split = data_name.split('_')
        if data_name_split[0] == 'vocal':
            continue
        else:
            num = data_name_split[1]
            category = data_name[12:-5]
            if category not in category_dict.keys():
                category_dict[category] = 0
            category_dict[category] += 1

            noise_data_name = data_name
            clean_data_name = 'vocal_%s.flac' % (num)

            noise_data_path = os.path.join(data_path, noise_data_name)
            out_noise, _ = torchaudio.load(noise_data_path)
            audio_length = out_noise.shape[1]

            data.append([noise_data_name, clean_data_name, str(audio_length)])
    for key, value in category_dict.items():
        category_list.append([str(key), str(value)])

    len_data = len(data)
    data = np.array(data)
    random_num_list = np.random.choice(len_data, len_data, replace=False)

    train_index = np.array(random_num_list[:int(len(random_num_list) * train_val_ratio)], dtype=int)
    val_index = np.array(random_num_list[int(len(random_num_list) * train_val_ratio):], dtype=int)
    train_data = data[train_index]
    valid_data = data[val_index]

    np.savetxt(os.path.join(save_path, 'train.csv'), train_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(save_path, 'val.csv'), valid_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(save_path, 'category.csv'), category_list, fmt='%s', delimiter=',')

    print('---------------Statistics!!----------------')
    print('# of training data: ', len(train_data))
    print('# of validation data: ', len(valid_data))
    print('# of category: ', len(category_list))

    

    
            

            
            