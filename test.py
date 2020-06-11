import torch
import numpy as np

def test_generate(generator, real, source, spec_transform=None, title="", writer=None, device="cpu"):
    noise = torch.randn(real.shape).to(device)
    fake = generator(torch.cat((source,noise), 1))
        
    real = torch.squeeze(real[0,:,:,:])
    fake = torch.squeeze(fake[0,:,:,:])
        
    if torch.cuda.is_available():
        real = real.cpu()
        fake = fake.cpu()
        
    real = real.detach().numpy()
    fake = fake.detach().numpy()
        
    if spec_transform is not None:
        fake = spec_transform(fake)
        
    fake_max = np.max(fake)
    if fake_max > 1: fake_max = np.maximum(2*np.percentile(fake, 0.95), 1e-16)
    fake_min = np.maximum(np.min(fake), 0)
    real_max = np.max(real)
    if real_max > 1: real_max = np.maximum(2*np.percentile(real, 0.95), 1e-16)
    real_min = np.maximum(np.min(real), 0)
    
    fake = (fake - fake_min) / (fake_max - fake_min)
    real = (real - real_min) / (real_max - real_min)
    
    fake_title = "FAKE " + title
    fake_title += " ~ min: %.6f max: %.6f" % (fake_min, fake_max)
    real_title = "REAL " + title
    real_title += " ~ min: %.6f max: %.6f" % (real_min, real_max)
    
    if writer is not None:
        writer.add_image(fake_title, fake, 0, dataformats='HW')
        writer.add_image(real_title, real, 0, dataformats='HW')