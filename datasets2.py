import os
import glob

import librosa
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

data_folder = 'data/'
data_file = 'initial_aa_dataset_pitchthresholded.hdf5'
data_path = data_folder+data_file

time_ratio = 8

def generate_frames(data):
    input_size = data.shape[1]
    time_frame = data.shape[0] * time_ratio
    overlap = time_frame // 2
    num_frames = int(np.ceil(input_size / overlap)) - 1
    padded = np.zeros((1, data.shape[0], overlap * (num_frames + 1)))
    padded[0,:,:data.shape[1]] = data
    frames = []
    i = 0
    start = 0
    while i < num_frames:
        frames.append(padded[:,:,start:start+time_frame])
        i += 1
        start += overlap
    return frames

def shift_scale(shift, scale, inverse=False):
    def helper(data):
        if not inverse:
            result = (data + shift) * scale
            #if np.max(result) > 1: print('too big')
            #if np.min(result) < -1: print('too small')
        else:
            result = (data / scale) - shift
        return result
    return helper

class SpecDataset(Dataset):
    def __init__(self, data_path=data_path,  spec_transform=None, f0_transform=None):
        super().__init__()
        self.data = []

        hdf5_file = h5py.File(data_path, 'r')
        for styles in hdf5_file:
            for datapoints in hdf5_file[styles]:
                mfsc_real = np.transpose(hdf5_file[hdf5_file[styles][datapoints].name+'/log_mfsc_real'])
                mfsc_synth = np.transpose(hdf5_file[hdf5_file[styles][datapoints].name+'/log_mfsc_synth'])
                f0 = np.transpose(hdf5_file[hdf5_file[styles][datapoints].name+'/f0'])
                #f0[np.nonzero(f0)] = 69+12*np.log2(f0[np.nonzero(f0)]/440)
                f0 = 69+12*np.log2(f0/440)
                f0 = np.repeat(f0, mfsc_synth.shape[0], axis=0)
                
                if spec_transform is not None: mfsc_real = spec_transform(mfsc_real)
                if spec_transform is not None: mfsc_synth = spec_transform(mfsc_synth)
                if f0_transform is not None: f0 = f0_transform(f0)
                    
                mfsc_frames_r = generate_frames(mfsc_real)
                mfsc_frames_s = generate_frames(mfsc_synth)
                f0_frames = generate_frames(f0)
               
                                          
                for mfsc_frame_r, mfsc_frame_s, f0_frame in zip(mfsc_frames_r, mfsc_frames_s, f0_frames):
                    features = dict()
                    features["mfsc_real"] = torch.from_numpy(mfsc_frame_r).float()
                    features["mfsc_synth"] = torch.from_numpy(mfsc_frame_s).float()
                    features["f0"] = torch.from_numpy(f0_frame).float()
                
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