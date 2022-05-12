from torch.utils.data import DataLoader
# from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from timm.data import ImageDataset
from timm.data.transforms_factory import create_transform


timm_transform = create_transform(224, scale=(0.7, 1.0), is_training=True, auto_augment='rand-mstd0.5')
NUM_WORKERS = 0
batch_size = 40
IMAGENET_STATS = ([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])


inference_transforms = transforms.Compose([
              transforms.Resize(size=256),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize(*IMAGENET_STATS)
        ])


class BirdsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize(*IMAGENET_STATS)
        ])
        self.transform = inference_transforms

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        ids = ImageDataset('data', transform=tfms)
        # index_to_name = {v: k for k, v in ids.parser.class_to_idx.items()}
        # import json
        # with open('index_to_name.json', 'w') as f:
        #     json.dump(index_to_name, f)

        targets = [c for (f, c) in ids.parser.samples]
        train_indices, val_indices = train_test_split(list(range(len(targets))), test_size=0.13,
                                                      stratify=targets, shuffle=True)
        self.train_dataset = Subset(ids, train_indices)
        self.train_dataset.transform = self.augmentation
        self.val_dataset = Subset(ids, val_indices)
        self.val_dataset.transform = self.transform

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        mnist_train = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        return mnist_test


birds = BirdsDataModule()
birds.prepare_data()
birds.setup()

samples = next(iter(birds.val_dataloader()))
