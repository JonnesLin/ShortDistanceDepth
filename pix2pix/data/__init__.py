"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from pix2pix.data.base_dataset import BaseDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "pix2pix.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
                dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    custom_dataset = CustomDatasetDataLoader(opt)
    dataset = custom_dataset.get_dataset()
    return dataset


def create_dataloader(opt, train=True):
    custom_dataset = CustomDatasetDataLoader(opt, train=train)
    data_loader = custom_dataset.get_loader()
    return data_loader


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, train=True):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        transform = None
        if train:
            transform = A.Compose(
                [
                    A.Resize(384, 384),
                    A.RandomBrightnessContrast(),
                    A.Rotate(limit=80),
                    A.augmentations.transforms.Flip(),
                    # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                    # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    # A.augmentations.geometric.rotate.SafeRotate (limit=90),
                    # A.augmentations.dropout.grid_dropout.GridDropout(ratio=0.2),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(transpose_mask=True),
                ]
            )
        else:
            transform = A.Compose(
                [
                    A.Resize(384, 384),
                    # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                    # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                    # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(transpose_mask=True),
                    # ToTensorV2(),
                ]
            )
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        if train:
            opt.phase = 'train'
            self.dataset = dataset_class(opt, transform=transform)
        else:
            opt.phase = 'test'
            self.dataset = dataset_class(opt, transform=transform)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=train,
            num_workers=int(opt.num_threads))

    def get_dataset(self):
        return self.dataset

    def get_loader(self):
        return self.dataloader

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
