import glob
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data


num_classes = 4
ignore_label = 255
path = 'ACDC-dataset/ACDC/2d'


def make_dataset(mode, root):
    if mode == "train":
        img_path = os.path.join(root, "image", "train")
        mask_path = os.path.join(root, "mask", "train")
    elif mode == "val":
        img_path = os.path.join(root, "image", "val")
        mask_path = os.path.join(root, "mask", "val")
    elif mode == "test":
        img_path = os.path.join(root, "image", "test")
        mask_path = os.path.join(root, "mask", "test")
    else:
        raise ValueError('Dataset split specified does not exist!')

    img_paths = [f for f in glob.glob(os.path.join(img_path, "*.png"))]
    items = []
    for im_p in img_paths:
        items.append((im_p, os.path.join(mask_path, os.path.basename(im_p)), os.path.basename(im_p)))
    return items


class ACDC_al(data.Dataset):
    def __init__(
        self,
        quality,
        mode,
        data_path='',
        joint_transform=None,
        sliding_crop=None,
        transform=None,
        target_transform=None,
        candidates_option=False,
        region_size=(64, 64),
        pretrain=False,
        num_each_iter=1,
        only_last_labeled=True,
        split='train',
        re_all_length=False,
    ):
        self.num_each_iter = num_each_iter
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = os.path.join(data_path, path)
        self.re_all_length = re_all_length
        self.imgs = make_dataset(mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.pretrain = pretrain
        self.supervised = False

        # Resolve split file robustly to avoid cwd issues in notebooks
        candidate_split_paths = [
            Path(__file__).resolve().parent / 'acdc_al_splits.npy',
            Path.cwd() / 'data' / 'acdc_al_splits.npy',
            Path(__file__).resolve().parent.parent / 'data' / 'acdc_al_splits.npy',
        ]
        splits_path = next((p for p in candidate_split_paths if p.exists()), None)
        if splits_path is None:
            searched = '\n'.join(str(p) for p in candidate_split_paths)
            raise FileNotFoundError(f'Missing splits file. Tried:\n{searched}')
        splits = np.load(str(splits_path), allow_pickle=True).item()
        self.state_subset = [img for img in self.imgs if img[-1] in splits['d_s']]
        self.state_subset_regions = {}
        for i in range(len(splits['d_s'])):
            x_r1 = np.arange(0, 256 - region_size[0] + 1, region_size[0])
            y_r1 = np.arange(0, 256 - region_size[1] + 1, region_size[1])
            self.state_subset_regions[i] = np.array(np.meshgrid(x_r1, y_r1)).T.reshape(-1, 2)

        if split == 'select_unlab_region':
            self.imgs = [img for img in self.imgs if img[-1] in splits['d_t']]
        elif split == 'pretrain':
            self.imgs = [img for img in self.imgs if img[-1] in splits['p_t']]
        elif split == 'train':
            splits_sum = splits['t_q'] + splits['p_t']
            self.imgs = [img for img in self.imgs if img[-1] in splits_sum]
        elif split == 'unlab':
            splits_sum = splits['d_t'] + splits['t_q']
            self.imgs = [img for img in self.imgs if img[-1] in splits_sum]
        elif split == 'unlab_final':
            self.imgs = [img for img in self.imgs if img[-1] in splits['d_t']]
        elif split == 'full_sup_final':
            splits_sum = splits['p_t'] + splits['t_q']
            self.imgs = [img for img in self.imgs if img[-1] in splits_sum]
        self.split = split

        self.end_al = False
        self.balance_cl = []
        self.only_last_labeled = only_last_labeled
        self.candidates = candidates_option
        self.selected_images = []
        self.selected_regions = dict()
        self.list_regions = []
        self.num_imgs = len(self.imgs)

        splitters_x = np.arange(0, 256 - region_size[0] + 1, region_size[0])
        splitters_y = np.arange(0, 256 - region_size[1] + 1, region_size[1])
        splitters_mesh = np.array(np.meshgrid(splitters_y, splitters_x)).T.reshape(-1, 2)
        prov_splitters = splitters_mesh.copy()
        prov_splitters_x = list(prov_splitters[:, 1])
        prov_splitters_y = list(prov_splitters[:, 0])
        self.unlabeled_regions_x = [deepcopy(prov_splitters_x) for _ in range(self.num_imgs)]
        self.unlabeled_regions_y = [deepcopy(prov_splitters_y) for _ in range(self.num_imgs)]
        self.num_unlabeled_regions_total = (256 * 256) // (region_size[0] * region_size[1]) * self.num_imgs
        self.region_size = region_size

    def get_subset_state(self, index):
        img_path, mask_path, im_name = self.state_subset[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img = img.resize((256, 256), Image.Resampling.BILINEAR)
        mask = mask.resize((256, 256), Image.Resampling.NEAREST)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask, None, (img_path, mask_path, im_name), self.state_subset_regions[index]

    def __getitem__(self, index):
        if self.pretrain:
            img_path, mask_path, im_name = self.imgs[index]
        else:
            if self.candidates or self.end_al:
                img_path, mask_path, im_name = self.imgs[self.selected_images[index]]
                selected_region_ind = np.random.choice(len(self.selected_regions[self.selected_images[index]]))
                selected_region = self.selected_regions[self.selected_images[index]][selected_region_ind]
                selected = [self.selected_images[index]]
            else:
                if self.only_last_labeled:
                    selected = self.list_regions[len(self.list_regions) - self.num_each_iter:][index]
                else:
                    selected = self.list_regions[index]

                img_path, mask_path, im_name = self.imgs[selected[0]]
                selected_region = selected[1]

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img = img.resize((256, 256), Image.Resampling.BILINEAR)
        mask = mask.resize((256, 256), Image.Resampling.NEAREST)
        mask = np.array(mask)

        if not (self.candidates or self.pretrain):
            mask = self.maskout_unselected_regions(mask, selected[0], self.region_size)

        mask = Image.fromarray(mask.astype(np.uint8))
        if self.joint_transform is not None:
            if not (self.candidates or self.pretrain or self.split in ('unlab', 'unlab_final')):
                img, mask = self.joint_transform(img, mask, selected_region)
            else:
                img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.split in ('unlab', 'unlab_final'):
            return img, 0, (img_path, 0, im_name)
        if self.pretrain:
            return img, mask, (img_path, mask_path, im_name)
        return img, mask, (img_path, mask_path, im_name), selected_region[0] if not self.candidates else \
            self.selected_images[index], 0

    def maskout_unselected_regions(self, mask, image, region_size=(64, 64)):
        masked = np.full(mask.shape, ignore_label)
        for region in self.selected_regions[image]:
            r_x = int(region[1])
            r_y = int(region[0])
            masked[r_x: r_x + region_size[1], r_y: r_y + region_size[0]] = mask[r_x: r_x + region_size[1],
                                                                           r_y: r_y + region_size[0]]
        return masked

    def get_specific_item(self, path):
        img_path, mask_path, im_name = self.imgs[path]
        cost_img = None
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img = img.resize((256, 256), Image.Resampling.BILINEAR)
        mask = mask.resize((256, 256), Image.Resampling.NEAREST)
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask, cost_img, (img_path, mask_path, im_name)

    def __len__(self):
        if self.candidates and self.supervised:
            return len(self.imgs)
        if self.re_all_length:
            return len(self.imgs)
        if self.candidates or self.end_al:
            return len(self.selected_images)
        if self.only_last_labeled:
            return self.num_each_iter
        return len(self.list_regions)

    def get_random_unlabeled_region_image(self, index):
        counter_i = int(np.random.choice(range(len(self.unlabeled_regions_x[index])), 1, replace=False))
        counter_x = self.unlabeled_regions_x[index].pop(counter_i)
        counter_y = self.unlabeled_regions_y[index].pop(counter_i)
        return counter_x, counter_y

    def labeled_set(self):
        return self.selected_images

    def get_labeled_regions(self):
        return self.selected_regions

    def get_unlabeled_regions(self):
        return deepcopy(self.unlabeled_regions_x), deepcopy(self.unlabeled_regions_y)

    def set_unlabeled_regions(self, rx, ry):
        self.unlabeled_regions_x = rx
        self.unlabeled_regions_y = ry

    def get_num_unlabeled_regions(self, region_size=128):
        return self.num_unlabeled_regions_total

    def get_num_unlabeled_regions_image(self, index):
        return len(self.unlabeled_regions_x[index])

    def get_num_labeled_regions(self):
        labeled_regions = 0
        for _, value in self.selected_regions.items():
            labeled_regions += len(value)
        return labeled_regions

    def get_candidates(self, num_regions_unlab=1000):
        unlabeled_regions = 0
        candidates = []
        images_list = list(range(self.num_imgs))
        while unlabeled_regions <= num_regions_unlab:
            if len(images_list) == 0:
                raise ValueError('There is no more unlabeled regions to fulfill the amount we want!')
            index = np.random.choice(len(images_list))
            candidate = images_list.pop(index)
            num_regions_left = self.get_num_unlabeled_regions_image(int(candidate))
            if num_regions_left > 0:
                unlabeled_regions += num_regions_left
                candidates.append(candidate)
        return candidates

    def check_class_region(self, img, region, region_size=(64, 64), eps=1E-7):
        img_path, mask_path, im_name = self.imgs[img]
        mask = Image.open(mask_path)
        mask = mask.resize((256, 256), Image.Resampling.NEAREST)
        mask = np.array(mask)
        r_x = int(region[1])
        r_y = int(region[0])
        region_classes = mask[r_x: r_x + region_size[1], r_y: r_y + region_size[0]]
        unique, counts = np.unique(region_classes, return_counts=True)
        balance = []
        for cl in range(0, self.num_classes + 1):
            if cl in unique:
                balance.append(counts[unique == cl].item() / counts.sum())
            else:
                balance.append(eps)
        self.balance_cl.append(balance)

    def add_index(self, paths, region=None):
        if isinstance(paths, list):
            for path in paths:
                if path not in self.selected_images:
                    self.selected_images.append(int(path))
                if region is not None:
                    if int(path) in self.selected_regions.keys():
                        if region not in self.selected_regions[int(path)]:
                            self.selected_regions[int(path)].append(region)
                            self.add_index_(path, region)
                    else:
                        self.selected_regions[int(path)] = [region]
                        self.add_index_(path, region)

        else:
            if paths not in self.selected_images:
                self.selected_images.append(int(paths))
            if region is not None:
                if int(paths) in self.selected_regions.keys():
                    if region not in self.selected_regions[int(paths)]:
                        self.selected_regions[int(paths)].append(region)
                        self.add_index_(paths, region)
                    else:
                        print('Region already added!')
                else:
                    self.selected_regions[int(paths)] = [region]
                    self.add_index_(paths, region)

    def add_index_(self, path, region):
        self.list_regions.append((int(path), region))
        self.num_unlabeled_regions_total -= 1

        self.check_class_region(int(path), (region[0], region[1]), self.region_size)
        for i in range(len(self.unlabeled_regions_x[int(path)])):
            if self.unlabeled_regions_x[int(path)][i] == region[0] and \
                    self.unlabeled_regions_y[int(path)][i] == region[1]:
                self.unlabeled_regions_x[int(path)].pop(i)
                self.unlabeled_regions_y[int(path)].pop(i)
                break

    def del_index(self, paths):
        self.selected_images.remove(paths)

    def reset(self):
        self.selected_images = []
