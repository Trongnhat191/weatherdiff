import os
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import random

class MyDataset:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation=''):
        train_dir = os.path.join(self.config.data.data_dir, 'train')
        val_dir = os.path.join(self.config.data.data_dir, 'test')

        train_dataset = CustomDataset(dir=train_dir,
                                      patch_size=self.config.data.image_size,
                                      n=self.config.training.patch_n,
                                      transforms=self.transforms,
                                      parse_patches=parse_patches)
        val_dataset = CustomDataset(dir=val_dir,
                                    patch_size=self.config.data.image_size,
                                    n=self.config.training.patch_n,
                                    transforms=self.transforms,
                                    parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True):
        super().__init__()
        self.dir = dir
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

        self.input_dir = os.path.join(dir, 'hazy')
        self.gt_dir = os.path.join(dir, 'GT')

        # Lấy danh sách tên file từ thư mục input
        self.image_files = sorted(os.listdir(self.input_dir))
        print(f"Found {len(self.image_files)} images in {self.input_dir}")
    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        input_path = os.path.join(self.input_dir, image_file)
        gt_path = os.path.join(self.gt_dir, image_file)

        try:
            input_img = PIL.Image.open(input_path).convert('RGB')
            gt_img = PIL.Image.open(gt_path).convert('RGB')
        except Exception as e:
            print(f"Errorrrrrrrrr loading image {image_file}: {e}")
            # Trả về một tensor rỗng hoặc xử lý lỗi khác nếu cần
            return torch.zeros(self.n, 6, self.patch_size, self.patch_size), "error_id"

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_crops = self.n_random_crops(input_img, i, j, h, w)
            gt_crops = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_crops[i]), self.transforms(gt_crops[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), image_file
        else:
            # Logic để xử lý ảnh nguyên vẹn (nếu cần)
            # ... (có thể giữ nguyên hoặc chỉnh sửa từ raindrop.py)
            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), image_file