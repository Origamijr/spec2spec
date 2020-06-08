
from models import *
from datasets import SpecDataset, get_dataloaders
import logging
import time
from tensorboardX import SummaryWriter
import torch
import os
from collections import defaultdict

def train(generator,
          discriminator,
          train_dataloader,
          val_dataloader = None,
          epochs=100, 
          lambda_pixel = 100
          ):

    # Bookkeeping setup
    ts = time.strftime('%Y_%b_%d_%H_%M_%S', time.gmtime())

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    writer = SummaryWriter(os.path.join("logs", repr(ts)))

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    
    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_pixelwise = criterion_pixelwise.cuda()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using %s" % device)

    print(generator)
    print(discriminator)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Weight initialization and scheduling?

    def train_discriminator(fake, fake_source, real):

        # setup
        for param in discriminator.parameters():
            param.requires_grad = True
        optimizer_D.zero_grad()
        # Fake
        fake_AB = torch.cat((fake_source, fake), 1)
        pred_fake = discriminator(fake_AB.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros(pred_fake.shape))
        # Real
        real_AB = torch.cat((fake_source, real), 1)
        pred_real = discriminator(real_AB)
        loss_D_real = criterion_GAN(pred_real, torch.ones(pred_real.shape))
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        # update
        optimizer_D.step()

        return float(loss_D.item())

    def train_generator(fake, fake_source, real):
        # setup
        for param in discriminator.parameters():
            param.requires_grad = False
        optimizer_G.zero_grad()
        # Fooling discriminator
        fake_AB = torch.cat((fake_source, fake), 1)
        pred_fake = discriminator(fake_AB.detach())
        loss_G_GAN = criterion_GAN(pred_fake, torch.ones(pred_fake.shape))
        # Reconstruction
        loss_G_L1 = criterion_pixelwise(fake, real) * lambda_pixel
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        # update
        optimizer_G.step()
        
        return float(loss_G_GAN.item()), float(loss_G_L1.item()), float(loss_G.item())

    # Main tranining loop
    for epoch in range(epochs):

        tracker = defaultdict(list)

        for i, batch in enumerate(train_dataloader):
            real = batch["spectrogram_real"].to(device)
            spec = batch["spectrogram_fake"].to(device)
            f0 = batch["f0"].to(device)
            fake_source = torch.cat((spec,f0), 1)

            # Create fakes
            fake = generator(fake_source)

            # Train Discriminator
            loss_D = train_discriminator(fake, fake_source, real)

            tracker['loss_D'].append(loss_D)

            # Train generator
            loss_G_GAN, loss_G_L1, loss_G = train_generator(fake, fake_source, real)
            

            # BOOKKEEPING

            tracker['loss_G_GAN'].append(loss_G_GAN)
            tracker['loss_G_L1'].append(loss_G_L1)
            tracker['loss_G'].append(loss_G)

            logger.info("TRAIN: Batch %d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D %9.4f"
                        %(i, len(train_dataloader)-1, loss_G_GAN, loss_G_L1, loss_G, loss_D))

        
        mean_loss_G_GAN = sum(tracker['loss_G_GAN']) / len(tracker['loss_G_GAN'])
        mean_loss_G_L1 = sum(tracker['loss_G_L1']) / len(tracker['loss_G_L1'])
        mean_loss_G = sum(tracker['loss_G']) / len(tracker['loss_G'])
        mean_loss_D = sum(tracker['loss_D']) / len(tracker['loss_D'])

        logger.info("TRAIN: Epoch %04d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D %9.4f"
                    %(epoch, epochs, mean_loss_G_GAN, mean_loss_G_L1, mean_loss_G, mean_loss_D))

        writer.add_scalar("Train-Epoch/loss_G_GAN", mean_loss_G_GAN, epoch)
        writer.add_scalar("Train-Epoch/loss_G_L1", mean_loss_G_L1, epoch)
        writer.add_scalar("Train-Epoch/loss_G", mean_loss_G, epoch)
        writer.add_scalar("Train-Epoch/loss_D", mean_loss_D, epoch)

        # Save model checkpoints
        if (epoch % 10 == 0):
            torch.save(generator.state_dict(), "saved_models/generator_%s_%d.pth" % (ts, epoch))
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%s_%d.pth" % (ts, epoch))


if __name__ == "__main__":
    generator = UnetGenerator(2, 1, 8)
    discriminator = NLayerDiscriminator(3)

    dataset = SpecDataset()
    train_dl, _ = get_dataloaders(dataset)

    train(generator, discriminator, train_dl)