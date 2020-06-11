import os
import glob

import librosa
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

data_folder = 'data/'
data_file = 'initial_aa_dataset.hdf5'
data_path = data_folder+data_file



def scale_shift(shift, scale, inverse=False):
    def helper(data):
    	if not inverse:
        	result = (data + shift) * scale
        	if np.max(result) > 1: print('too big')
        	if np.min(result) < -1: print('too small')
        else:
        	result = (data/scale) - shift
        return result
    return helper


class SpecDataset(Dataset):
    def __init__(self, data_path=data_path,  spec_transform=None, f0_transform=None):
        super().__init__()
        self.data = []

        hdf5_file = h5py.File(data_path, 'r')
        for styles in hdf5_file:
    		for datapoints in hdf5_files[styles]:
        		f0 = np.transpose(hdf5_files[hdf5_files[styles][datapoints].name+'/f0'])
        		mfsc_synth = np.transpose(hdf5_files[hdf5_files[styles][datapoints].name+'/log_mfsc_synth'])
        		mfsc_real = np.transpose(hdf5_files[hdf5_files[styles][datapoints].name+'/log_mfsc_real'])
        		f0 = np.repeat(f0, spec_synth.shape[0], axis=0)

            	if f0_transform is not None: f0 = f0_transform(f0)
            	if spec_transform is not None: mfsc_synth = spec_transform(mfsc_synth)
            	if spec_transform is not None: mfsc_real = spec_transform(mfsc_real)
            	
            	features = dict()
                features["mfsc_real"] = torch.from_numpy(mfsc_real).float()
                features["mfsc_synth"] = torch.from_numpy(mfsc_synth).float()
                features["f0"] = torch.from_numpy(f0).float()
            	
            	self.data.append(features)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

"""
Get split data loaders
"""
def get_dataloaders(dataset,
                    spec_transform=None, 
                    f0_transform=None, 
                    batch_size=1, 
                    shuffle=True,
                    split=1,
                    num_workers=0):
    
    ds = dataset
    trn_len = int(len(ds) * split)
    val_len = len(ds) - trn_len
    trn, val = random_split(ds, [trn_len, val_len])
    
    trn_dl = DataLoader(trn,
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers)
    val_dl = None
    if split < 1:
        val_dl = DataLoader(val, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers)

    return trn_dl, val_dl

if __name__ == "__main__":
    #process_audio_files()
    pass