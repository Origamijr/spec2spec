import os
import glob

import librosa
import pyworld as pw
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

data_folder = 'data/'
real_audio = data_folder + 'real_audio/'
fake_audio = data_folder + 'fake_audio/'
real_data = data_folder + 'real/'
fake_data = data_folder + 'fake/'

time_ratio = 2

"""
Generates preprocessed hdf5 files
"""
def process_audio_files():
    import os
    # iterate over audio files
    for file in os.listdir(real_audio): # Target audio
        create_data_from_audio(real_audio + file, real_data + os.path.splitext(file)[0] + '.hdf5')
    for file in os.listdir(fake_audio): # Source audio
        create_data_from_audio(fake_audio + file, fake_data + os.path.splitext(file)[0] + '.hdf5')
    pass

"""
Process a single file
"""
def create_data_from_audio(audio_file, out_file):
    y, sr = librosa.load(audio_file)
    x = y.astype(np.float64)
    f0, timeaxis = pw.harvest(x, sr, frame_period=10)   # Extract f0
    spec = pw.cheaptrick(x, f0, timeaxis, sr).T         # Get spectrogram (idk if this is mel)
    f0 = np.reshape(f0, (1, spec.shape[1]))
    f0_mat = np.repeat(f0, spec.shape[0], axis=0)       # Repeat f0 accross spectrogram

    # Store preprocessed data in hdf5 file
    with h5py.File(out_file, mode='a') as hdf5_file:
        hdf5_file.create_dataset("spec", spec.shape, np.float32)
        hdf5_file.create_dataset("f0", f0_mat.shape, np.float32)
        hdf5_file["spec"][:,:] = spec.astype(np.float32)
        hdf5_file["f0"][:,:] = f0_mat.astype(np.float32)

    print('wrote %s' % out_file)

"""
Creates overlapping fixed sized frames of the data
"""
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

def overlapadd_frames(frames):
    #TODO
    pass

def log_scale(data):
    return np.log(data + 1e-10)

def real_scale(data):
    return np.exp(data) - 1e-10

"""
Iterable dataset for training
"""
class SpecDataset(Dataset):
    def __init__(self, data_dir=data_folder,  spec_transform=None, f0_transform=None):
        super().__init__()
        self.data = []

        real_data = data_folder + 'real/'
        fake_data = data_folder + 'fake/'

        # Pair data (hopefully the names of the files are correlated)
        for real_file, fake_file in zip(os.listdir(real_data).sort(), os.listdir(fake_data).sort()):
            print('paired %s and %s' % (real_file, fake_file))

            # Real data (target)
            spec_r, f0_r = __read_hdf5(real_data + real_file)
            if spec_transform is not None: spec_r = spec_transform(spec_r)
            if f0_transform is not None: f0_r = f0_transform(f0_r)
            spec_frames_r = generate_frames(spec_r)
            f0_frames_r = generate_frames(f0_r)

            # Fake data (source)
            spec_f, f0_f = __read_hdf5(fake_data + fake_file)
            if spec_transform is not None: spec_f = spec_transform(spec_f)
            if f0_transform is not None: f0_f = f0_transform(f0_f)
            spec_frames_f = generate_frames(spec_f)
            f0_frames_f = generate_frames(f0_f)

            # Iterate over fixed sized frames
            for spec_frame_r, f0_frame_r, spec_frame_f, f0_frame_f in zip(spec_frames_r, f0_frames_r, spec_frames_f, f0_frames_f):
                features = dict()
                features["spectrogram_real"] = torch.from_numpy(spec_frame_r).float()
                features["spectrogram_fake"] = torch.from_numpy(spec_frame_f).float()
                features["f0"] = torch.from_numpy(f0_frame_r).float()
                self.data.append(features)

    def __read_hdf5(self, file):
        with h5py.File(file, mode='a') as hdf5_file:
            spec = hdf5_file["spec"][:,:]
            f0 = hdf5_file["f0"][:,:]
            return spec, f0

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
                    batch_size=5, 
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
    val_dl = DataLoader(val, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers)

    return trn_dl, val_dl

if __name__ == "__main__":
    #process_audio_files()