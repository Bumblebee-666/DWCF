import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch


class MOSEI_dataset(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False,
                 label_protocol='paper_threshold'):
        super(MOSEI_dataset, self).__init__()
        # 兼容 Inforeg 的路径结构，假设数据在 Inforeg/data/MOSEI 下
        candidate_paths = [
            os.path.join(dataset_path, data + '_data.pkl' if if_align else data + '_data_noalign.pkl'),
            os.path.join(dataset_path, data + '.pkl'),
            os.path.join(dataset_path, data),
        ]
        dataset_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if dataset_path is None:
            raise FileNotFoundError(f"Dataset not found in candidates: {candidate_paths}")

        dataset = pickle.load(open(dataset_path, 'rb'))
        actual_split = split_type
        if actual_split not in dataset:
            if actual_split == 'valid' and 'dev' in dataset:
                actual_split = 'dev'
            elif actual_split == 'dev' and 'valid' in dataset:
                actual_split = 'valid'
            else:
                raise KeyError(f"Split '{split_type}' not found in dataset keys: {list(dataset.keys())}")

        split_data = dataset[actual_split]

        # 提取特征
        self.vision = torch.tensor(split_data['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(split_data['text'].astype(np.float32)).cpu().detach()
        self.audio = split_data['audio'].astype(np.float32)
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio_finfo = np.finfo(np.float32)
        audio_extreme_mask = (self.audio <= audio_finfo.min / 2) | (self.audio >= audio_finfo.max / 2)
        if np.any(audio_extreme_mask):
            self.audio = self.audio.copy()
            self.audio[audio_extreme_mask] = 0.0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.label_protocol = label_protocol
        if label_protocol == 'classification' and 'classification_labels' in split_data:
            self.labels = torch.tensor(split_data['classification_labels'].astype(np.int64)).cpu().detach()
            self.has_classification_labels = True
        elif 'labels' in split_data:
            self.labels = torch.tensor(split_data['labels'].astype(np.float32)).cpu().detach()
            self.has_classification_labels = False
        elif 'regression_labels' in split_data:
            self.labels = torch.tensor(split_data['regression_labels'].astype(np.float32)).cpu().detach()
            self.has_classification_labels = False
        elif 'classification_labels' in split_data:
            self.labels = torch.tensor(split_data['classification_labels'].astype(np.int64)).cpu().detach()
            self.has_classification_labels = True
        else:
            raise KeyError(f"No supported label field found in split '{actual_split}'. Keys: {list(split_data.keys())}")

        self.data = data
        self.split_type = actual_split
        self.n_modalities = 3

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.has_classification_labels:
            target = int(self.labels[index].reshape(-1)[0].item())
        else:
            Y = float(self.labels[index].reshape(-1)[0].item())

            # 将连续值标签转换为三分类（和 mosei 项目一致）
            threshold = 0.5
            if -3 <= Y <= -threshold:
                target = 0
            elif -threshold < Y < threshold:
                target = 1
            else:
                target = 2

        # 返回格式尽量向 Inforeg 靠拢
        # Inforeg AVDataset 返回: spectrogram, image_n, label, av_file
        # 这里返回: text, audio, vision, target
        return self.text[index], self.audio[index], self.vision[index], target
