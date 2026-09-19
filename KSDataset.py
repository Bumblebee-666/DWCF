import copy
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def inv_norm_tensor(img):
    inv_norm = UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img[:, :, :] = inv_norm(img)
    return img


class KS_dataset(Dataset):
    def __init__(self, args, mode, select_ratio=1, v_norm=True, a_norm=False, name="KS"):
        self.args = args
        self.data = []
        self.label = []
        self.num_frames = max(1, int(getattr(args, 'use_video_frames', 3))) if args is not None else 3
        # ===== KS 相对 baseline（3）的关键新增 start: 测试阶段多视角评估 =====
        # baseline（3）里 KS 只有单视角 test，这里把测试视角数和 resize 显式参数化。
        # 这样验证/测试时可以对同一个样本做多次稳定取帧与裁剪，再平均 logits，
        # 这是这次 KS 能把 Visual Acc 稳住并顶上去的关键步骤之一。
        self.eval_num_views = max(1, int(getattr(args, 'ks_eval_num_views', 1))) if args is not None else 1
        self.eval_resize = max(224, int(getattr(args, 'ks_eval_resize', 256))) if args is not None else 256
        # ===== KS 相对 baseline（3）的关键新增 end: 测试阶段多视角评估 =====

        data_root = getattr(args, 'ks_data_root', './data/KineticSound')
        if mode == 'train':
            csv_path = os.path.join(data_root, 'my_train_fixed.txt')
            self.audio_path = os.path.join(data_root, 'train_spec')
            self.visual_path = os.path.join(
                data_root, 'train-videos', 'train-set-img', 'Image-01-FPS'
            )
        else:
            csv_path = os.path.join(data_root, 'my_test_fixed.txt')
            self.audio_path = os.path.join(data_root, 'test_spec')
            self.visual_path = os.path.join(
                data_root, 'test-videos', 'test-set-img', 'Image-01-FPS'
            )

        with open(csv_path) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")
                sample_name = item[0]

                # ===== KS 相对 baseline（3）的关键新增 =====
                # baseline（3）对 KS 样本过滤更粗，这里显式检查：
                # 1) 音频是否存在；2) 视觉目录是否存在；3) 可用 jpg 是否足够。
                # 这样能尽量避免坏样本/空目录把视觉分支拖偏。
                if os.path.exists(os.path.join(self.audio_path, sample_name + '.npy')):
                    visual_dir = os.path.join(self.visual_path, sample_name)
                    if not os.path.isdir(visual_dir):
                        continue
                    jpg_files = [frame_name for frame_name in os.listdir(visual_dir) if frame_name.endswith('.jpg')]
                    if len(jpg_files) >= min(3, self.num_frames):
                        self.data.append(sample_name)
                        self.label.append(int(item[-1]))

        print('data load finish')
        self.normalize = v_norm
        self.mode = mode
        self._init_atransform()

        if mode == 'train' and select_ratio < 1:
            self._random_choice(select_ratio)
        print('# of files = %d ' % len(self.data))

    def _random_choice(self, select_ratio=0.40):
        num_data = len(self.data)
        selected_id = np.random.choice(np.arange(num_data), int(num_data * select_ratio))
        selected_data = [self.data[idx] for idx in selected_id]
        selected_label = [self.label[idx] for idx in selected_id]
        self.data = selected_data
        self.label = selected_label

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def _sorted_jpg_files(self, path):
        jpg_files = [file_name for file_name in os.listdir(path) if file_name.endswith('.jpg')]

        def sort_key(file_name):
            stem = os.path.splitext(file_name)[0]
            try:
                return (0, int(stem))
            except ValueError:
                return (1, stem)

        return sorted(jpg_files, key=sort_key)

    def _select_frame_files(self, jpg_files, pick_num=None, view_idx=0, num_views=1):
        if pick_num is None:
            pick_num = self.num_frames
        file_num = len(jpg_files)
        if file_num == 0:
            return []

        # ===== KS 相对 baseline（3）的关键新增 =====
        # baseline（3）更像是“按规则拼文件名去猜帧”，这里改成：
        # 先基于真实存在的 jpg 列表分段，再从每段中选帧。
        # 训练阶段仍保留随机性；测试阶段则按 view_idx 做确定性采样，
        # 为后面的多视角评估提供稳定输入。
        boundaries = np.linspace(0, file_num, num=pick_num + 1, dtype=int)
        selected_files = []

        for index in range(pick_num):
            start = int(boundaries[index])
            end = int(boundaries[index + 1])
            if end <= start:
                end = min(file_num, start + 1)

            if self.mode == 'train':
                selected_idx = random.randint(start, end - 1) if end - start > 1 else start
            else:
                if num_views <= 1:
                    offset = 0.5
                else:
                    offset = (view_idx + 0.5) / float(num_views)
                position = start + (end - start) * offset
                selected_idx = int(np.clip(np.floor(position), 0, file_num - 1))

            selected_files.append(jpg_files[selected_idx])

        return selected_files

    def _crop_box(self, width, height, crop_h, crop_w, crop_idx):
        max_top = max(0, height - crop_h)
        max_left = max(0, width - crop_w)
        if crop_idx == 0:
            return (0, 0, crop_w, crop_h)
        if crop_idx == 1:
            return (max_left, 0, max_left + crop_w, crop_h)
        if crop_idx == 2:
            return (0, max_top, crop_w, max_top + crop_h)
        if crop_idx == 3:
            return (max_left, max_top, max_left + crop_w, max_top + crop_h)
        top = max_top // 2
        left = max_left // 2
        return (left, top, left + crop_w, top + crop_h)

    def _build_eval_view_tensors(self, path, jpg_files, normalize):
        # ===== KS 相对 baseline（3）的关键新增 =====
        # baseline（3）没有这一步。这里专门为 KS 测试构造多视角输入：
        # 多个时间采样视角 + 不同 crop / flip，再在 main.py 里做 logits 平均。
        # 这一步主要服务于 KS 的 Visual Acc 稳定提升。
        selected_per_view = [
            self._select_frame_files(jpg_files, pick_num=self.num_frames, view_idx=view_idx, num_views=self.eval_num_views)
            for view_idx in range(self.eval_num_views)
        ]
        to_tensor = transforms.ToTensor()
        normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        crop_size = 224
        max_base_views = 5
        tensors = []

        for view_idx, selected_files in enumerate(selected_per_view):
            crop_idx = view_idx % max_base_views
            use_flip = view_idx >= max_base_views
            frame_tensors = []

            for file_name in selected_files:
                full_path = os.path.join(path, file_name)
                img = Image.open(full_path).convert('RGB')
                img = img.resize((self.eval_resize, self.eval_resize), Image.BILINEAR)
                if use_flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                left, top, right, bottom = self._crop_box(self.eval_resize, self.eval_resize, crop_size, crop_size, crop_idx)
                img = img.crop((left, top, right, bottom))
                tensor_img = to_tensor(img)
                if normalize:
                    tensor_img = normalizer(tensor_img)
                frame_tensors.append(tensor_img.unsqueeze(1).float())

            tensors.append(torch.cat(frame_tensors, dim=1))

        return torch.stack(tensors, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]
        spectrogram = np.load(os.path.join(self.audio_path, av_file + '.npy'))

        path = os.path.join(self.visual_path, av_file)
        # ===== KS 相对 baseline（3）的关键新增 =====
        # baseline（3）在 KS 上没有先拿“真实存在且有序的帧列表”再采样，
        # 这里统一先排序 jpg_files，再交给 _select_frame_files 做分段取帧。
        jpg_files = self._sorted_jpg_files(path)

        if self.mode == 'train':
            transf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transf = [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ]
        if self.normalize:
            transf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        transf = transforms.Compose(transf)

        # ===== KS 相对 baseline（3）的关键新增 =====
        # 只在验证/测试阶段启用多视角评估；训练阶段仍保持单视图训练，
        # 这样不会明显拉长训练，但能显著减少 KS 测试时的视觉波动。
        if self.mode != 'train' and self.eval_num_views > 1:
            image_n = self._build_eval_view_tensors(path, jpg_files, normalize=self.normalize)
            label = self.label[idx]
            return spectrogram, image_n, label

        image = []
        image_arr = []
        selected_files = self._select_frame_files(jpg_files, pick_num=self.num_frames)
        for index, file_name in enumerate(selected_files):
            full_path = os.path.join(path, file_name)

            if os.path.exists(full_path):
                img = Image.open(full_path).convert('RGB')
            elif len(jpg_files) > 0:
                fallback_path = os.path.join(path, jpg_files[min(index, len(jpg_files) - 1)])
                img = Image.open(fallback_path).convert('RGB')
            else:
                img = Image.new('RGB', (224, 224))

            image.append(img)
            image_arr.append(transf(image[index]))
            image_arr[index] = image_arr[index].unsqueeze(1).float()

            if index == 0:
                image_n = copy.copy(image_arr[index])
            else:
                image_n = torch.cat((image_n, image_arr[index]), 1)

        label = self.label[idx]
        return spectrogram, image_n, label
