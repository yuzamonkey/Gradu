import os
import pickle
from natsort import natsorted
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset2D(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)
        return image, target

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True),
])


IMAGES_PATH = 'data/BrainMRISegmentation/kaggle_3m/'
DATAPATH = 'data/2D_segmentation_dataset'
SANITY_DATAPATH = 'data/2D_segmentation_sanity_dataset'

def load_dataset():
    if os.path.exists(DATAPATH):
        print('Loading 2D segmentation dataset...')
        data_file = open(DATAPATH, 'rb')
        data = pickle.load(data_file)
        data_file.close()
        return data
    else:
        print('Dataset is missing, creating 2D datasets...')
        _create_dataset_files()
        return load_dataset()

def load_sanity_dataset():
    if os.path.exists(DATAPATH) and os.path.exists(SANITY_DATAPATH):
        print('Loading 2D sanity dataset...')
        sanity_file = open(SANITY_DATAPATH, 'rb')
        sanity_data = pickle.load(sanity_file)
        sanity_file.close()
        return sanity_data
    else:
        print('Dataset is missing, creating 2D datasets...')
        _create_dataset_files()
        return load_sanity_dataset()

def _create_dataset_files():
    print('Iterating over directories containing images...')
    images = []
    targets = []
    for patient in os.listdir(IMAGES_PATH):
        patient_path = os.path.join(IMAGES_PATH, patient)
        for img in natsorted(os.listdir(patient_path)):
            im = plt.imread(os.path.join(patient_path, img))
            if im.ndim == 3:
                images.append(im)
            elif im.ndim == 2:
                targets.append(im)
            else:
                raise Exception()

    print('Creating dataobjects...')
    data = SegmentationDataset2D(
        images,
        targets,
        transform)

    # Arbitrary image to use in sanity test
    sanity_data = SegmentationDataset2D(
        [images[147]],
        [targets[147]],
        transform)

    print('Dumping objects to file...')
    data_file = open(DATAPATH, 'wb')
    sanity_file = open(SANITY_DATAPATH, 'wb')

    pickle.dump(data, data_file)
    pickle.dump(sanity_data, sanity_file)

    data_file.close()
    sanity_file.close()
    print('Done')