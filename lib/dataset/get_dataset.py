import os
import random
import numpy as np
from randaugment import RandAugment
import torchvision.transforms as transforms
from PIL import ImageDraw
from dataset.handlers import COCO2014_handler, VOC2012_handler, NUS_WIDE_handler, CUB_200_2011_handler


HANDLER_DICT = {
    'voc': VOC2012_handler,
    'coco': COCO2014_handler,
    'nus': NUS_WIDE_handler,
    'cub': CUB_200_2011_handler,
}


def get_datasets(args):
    train_transform = TransformUnlabeled_WS(args)

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()])

    # load data:
    source_data = load_data(args.dataset_dir)
	
	# generate indices to split official train set into train and val:
	# this setting follows paper "single positive"
    split_idx = {}
    (split_idx['train'], split_idx['val']) = generate_split(
		len(source_data['train']['images']),
		0.2,
		np.random.RandomState(1200)
		)
	

    ss_rng = np.random.RandomState(1200)

    for phase in ['train', 'val']:
        num_initial = len(split_idx[phase])
        num_final = int(np.round(1.0 * num_initial))
        split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]


    data_handler = HANDLER_DICT[args.dataset_name]
    train_dataset = data_handler(source_data['train']['images'][split_idx['train']], source_data['train']['labels_obs'][split_idx['train'], :], args.dataset_dir, transform=train_transform)
    val_dataset = data_handler(source_data['train']['images'][split_idx['val']], source_data['train']['labels'][split_idx['val'], :], args.dataset_dir, transform=val_transform)
    test_dataset = data_handler(source_data['val']['images'], source_data['val']['labels'], args.dataset_dir, transform=test_transform)

    return train_dataset, val_dataset, test_dataset

def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are 
    returned because this function is for splitting, not shuffling. 
    '''
    
    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    
    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])
    
    return (idx_1, idx_2)

def load_data(base_path):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['labels_obs'] = np.load(os.path.join(base_path, 'formatted_{}_labels_obs.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
    return data

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

class TransformUnlabeled_WS(object):
    def __init__(self, args):
        self.weak = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.Resize((args.img_size, args.img_size)),
			transforms.ToTensor()])

        self.strong = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.Resize((args.img_size, args.img_size)),
			CutoutPIL(cutout_factor=0.5),
			RandAugment(),
			transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
			transforms.ToTensor()])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong