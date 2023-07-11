from torch.utils import data

from data.nyu import get_NYUv2_dataset

"""
Preparation of dataloaders for Datasets
"""

def get_dataloader(dataset_name, 
                   path,
                   split='train', 
                   resolution='full',
                   augmentation='alhashim',
                   interpolation='linear', 
                   batch_size=8,
                   workers=4):
    if dataset_name == 'nyu':
        dataset = get_NYUv2_dataset(path, 
                split, 
                resolution=resolution)
    else:
        print('Dataset not existant')
        exit(0)

    dataloader = data.DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=(split=='train'),
            num_workers=workers, 
            pin_memory=True)
    return dataloader
