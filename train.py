
from models import *
from datasets2 import SpecDataset, get_dataloaders, shift_scale
import logging
import time
from tensorboardX import SummaryWriter
import torch
import os
from collections import defaultdict
from test import test_generate

def train(generator,
          discriminator,
          train_dataloader,
          val_dataloader = None,
          epochs=100, 
          lambda_pixel = 100,
          wgan=True,
          d_iterations=5,
          lambda_gp=0.1,
          useNoise=False
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

    if not wgan: d_iterations = 1
    
    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_pixelwise = criterion_pixelwise.cuda()
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using %s" % device)

    #print(generator)
    #print(discriminator)

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
        print(pred_fake.shape)
        loss_D_fake = torch.mean(pred_fake) if wgan else criterion_GAN(pred_fake, torch.zeros(pred_fake.shape).to(device))
        # Real
        real_AB = torch.cat((fake_source, real), 1)
        pred_real = discriminator(real_AB)
        loss_D_real = -torch.mean(pred_real) if wgan else criterion_GAN(pred_real, torch.ones(pred_real.shape).to(device))
        if wgan:
            # Gradient penalty (alternative to clipping)
            grad_penalty, _ = cal_gradient_penalty(discriminator, real_AB, fake_AB, device, lambda_gp=lambda_gp)
            grad_penalty.backward(retain_graph=True)
            # combine loss and calculate gradients
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward()
            # update
            optimizer_D.step()
    
            return float(loss_D.item()) + float(grad_penalty.item()), float(loss_D.item())
        else:
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
        loss_G_GAN = -torch.mean(pred_fake) if wgan else criterion_GAN(pred_fake, torch.ones(pred_fake.shape).to(device))
        # Reconstruction
        loss_G_L1 = criterion_pixelwise(fake, real) * lambda_pixel
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        # update
        optimizer_G.step()
        
        return float(loss_G_GAN.item()), float(loss_G_L1.item()), float(loss_G.item())

    def evaluate_model(fake_source, real):
        # Create fakes
        if useNoise:
            noise = torch.randn(spec.shape).to(device)
            fake = generator(torch.cat((fake_source,noise), 1))
        else:
            fake = generator(fake_source)
        
        for param in discriminator.parameters():
            param.requires_grad = False
            
        # Get discriminator results on fakes
        fake_AB = torch.cat((fake_source, fake), 1)
        pred_fake = discriminator(fake_AB.detach())
        loss_D_fake = torch.mean(pred_fake) if wgan else criterion_GAN(pred_fake, torch.zeros(pred_fake.shape).to(device))
        loss_G_GAN = -torch.mean(pred_fake) if wgan else criterion_GAN(pred_fake, torch.ones(pred_fake.shape).to(device))
        
        # Get discriminator results on reals
        real_AB = torch.cat((fake_source, real), 1)
        pred_real = discriminator(real_AB)
        loss_D_real = -torch.mean(pred_real) if wgan else criterion_GAN(pred_real, torch.ones(pred_real.shape).to(device))
        
        # Reconstruction
        loss_G_L1 = criterion_pixelwise(fake, real) * lambda_pixel
        
        # Gradient penalty
        if wgan: grad_penalty, _ = cal_gradient_penalty(discriminator, real_AB, fake_AB, device, lambda_gp=lambda_gp)
            
        # Combine losses
        loss_D = loss_D_fake + loss_D_real
        loss_G = loss_G_GAN + loss_G_L1
        
        if wgan:
            return float(loss_G_GAN.item()), float(loss_G_L1.item()), float(loss_G.item()), float(grad_penalty.item()), float(loss_D.item()) + float(grad_penalty.item())
        else:
            return float(loss_G_GAN.item()), float(loss_G_L1.item()), float(loss_G.item()), float(loss_D.item()) * 0.5

    # Main tranining loop
    for epoch in range(epochs):

        tracker = defaultdict(list)

        for i, batch in enumerate(train_dataloader):
            real = batch["mfsc_real"].to(device)
            spec = batch["mfsc_synth"].to(device)
            f0 = batch["f0"].to(device)
            fake_source = torch.cat((spec,f0), 1)

            # Create fakes
            if useNoise:
                noise = torch.randn(spec.shape).to(device)
                fake = generator(torch.cat((fake_source,noise), 1))
            else:
                fake = generator(fake_source)

            # Train Discriminator
            if wgan:
                loss_D = 0
                if wgan: loss_D_GAN = 0
                for j in range(d_iterations):
                    loss_D_t, loss_D_GAN_t = train_discriminator(fake, fake_source, real)
                    loss_D += loss_D_t
                    loss_D_GAN += loss_D_GAN_t
                loss_D /= d_iterations
                loss_D_GAN /= d_iterations
            else:
                loss_D = train_discriminator(fake, fake_source, real)

            # Train generator
            loss_G_GAN, loss_G_L1, loss_G = train_generator(fake, fake_source, real)
            

            # BOOKKEEPING

            tracker['loss_G_GAN'].append(loss_G_GAN)
            tracker['loss_G_L1'].append(loss_G_L1)
            tracker['loss_G'].append(loss_G)
            if wgan: tracker['loss_D_GAN'].append(loss_D_GAN)
            tracker['loss_D'].append(loss_D)

            if i % 10 == 0:
                if wgan:
                    logger.info("TRAIN: Batch %d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D_GAN %9.4f, loss_D %9.4f"
                                %(i, len(train_dataloader)-1, loss_G_GAN, loss_G_L1, loss_G, loss_D_GAN, loss_D))
                else:
                    logger.info("TRAIN: Batch %d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D %9.4f"
                                %(i, len(train_dataloader)-1, loss_G_GAN, loss_G_L1, loss_G, loss_D))
        
        mean_loss_G_GAN = sum(tracker['loss_G_GAN']) / len(tracker['loss_G_GAN'])
        mean_loss_G_L1 = sum(tracker['loss_G_L1']) / len(tracker['loss_G_L1'])
        mean_loss_G = sum(tracker['loss_G']) / len(tracker['loss_G'])
        if wgan: mean_loss_D_GAN = sum(tracker['loss_D_GAN']) / len(tracker['loss_D_GAN'])
        mean_loss_D = sum(tracker['loss_D']) / len(tracker['loss_D'])

        if wgan:
            logger.info("TRAIN: Epoch %d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D_GAN %9.4f, loss_D %9.4f"
                        %(epoch, epochs, mean_loss_G_GAN, mean_loss_G_L1, mean_loss_G, mean_loss_D_GAN, mean_loss_D))
        else:
            logger.info("TRAIN: Epoch %d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D %9.4f"
                        %(epoch, epochs, mean_loss_G_GAN, mean_loss_G_L1, mean_loss_G, mean_loss_D))

        writer.add_scalar("Train-Epoch/loss_G_GAN", mean_loss_G_GAN, epoch)
        writer.add_scalar("Train-Epoch/loss_G_L1", mean_loss_G_L1, epoch)
        writer.add_scalar("Train-Epoch/loss_G", mean_loss_G, epoch)
        if wgan: writer.add_scalar("Train-Epoch/loss_D_GAN", mean_loss_D_GAN, epoch)
        writer.add_scalar("Train-Epoch/loss_D", mean_loss_D, epoch)

        # Save model checkpoints
        if (epoch % 10 == 0):
            torch.save(generator.state_dict(), "saved_models/generator_%s_%d.pth" % (ts, epoch))
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%s_%d.pth" % (ts, epoch))
        
        # Test Generation
        if (epoch % 10 == 0):
            for i, batch in enumerate(train_dataloader):
                if i == 1:
                    real = batch["mfsc_real"].to(device)
                    spec = batch["mfsc_synth"].to(device)
                    f0 = batch["f0"].to(device)
                    fake_source = torch.cat((spec,f0), 1)
                    test_generate(generator, real, fake_source, spec_transform=shift_scale(7, 0.1, inverse=True), title=("TRAIN epoch %d" % epoch), writer=writer, device=device, useNoise=useNoise)
            
        if val_dataloader is not None:
            for i, batch in enumerate(val_dataloader):
                real = batch["mfsc_real"].to(device)
                spec = batch["mfsc_synth"].to(device)
                f0 = batch["f0"].to(device)
                fake_source = torch.cat((spec,f0), 1)
                
                # Evaluate            
                if wgan: loss_G_GAN, loss_G_L1, loss_G, loss_D, loss_D_GAN = evaluate_model(fake_source, real)
                else: loss_G_GAN, loss_G_L1, loss_G, loss_D = evaluate_model(fake_source, real)
                
                tracker['loss_G_GAN_v'].append(loss_G_GAN)
                tracker['loss_G_L1_v'].append(loss_G_L1)
                tracker['loss_G_v'].append(loss_G)
                if wgan: tracker['loss_D_GAN_v'].append(loss_D_GAN)
                tracker['loss_D_v'].append(loss_D)
    
                if i % 10 == 0:
                    if wgan:
                        logger.info("VALID: Batch %d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D_GAN %9.4f, loss_D %9.4f"
                                    %(i, len(train_dataloader)-1, loss_G_GAN, loss_G_L1, loss_G, loss_D_GAN, loss_D))
                    else:
                        logger.info("VALID: Batch %d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D %9.4f"
                                    %(i, len(train_dataloader)-1, loss_G_GAN, loss_G_L1, loss_G, loss_D))
            
            mean_loss_G_GAN_v = sum(tracker['loss_G_GAN_v']) / len(tracker['loss_G_GAN_v'])
            mean_loss_G_L1_v = sum(tracker['loss_G_L1_v']) / len(tracker['loss_G_L1_v'])
            mean_loss_G_v = sum(tracker['loss_G_v']) / len(tracker['loss_G_v'])
            if wgan: mean_loss_D_GAN_v = sum(tracker['loss_D_GAN_v']) / len(tracker['loss_D_GAN_v'])
            mean_loss_D_v = sum(tracker['loss_D_v']) / len(tracker['loss_D_v'])
    
            if wgan:
                logger.info("VALID: Epoch %d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D_GAN %9.4f, loss_D %9.4f"
                            %(epoch, epochs, mean_loss_G_GAN, mean_loss_G_L1, mean_loss_G, mean_loss_D_GAN, mean_loss_D))
            else:
                logger.info("VALID: Epoch %d/%i, loss_G_GAN %9.4f, loss_G_L1 %9.4f, loss_G %9.4f, loss_D %9.4f"
                            %(epoch, epochs, mean_loss_G_GAN, mean_loss_G_L1, mean_loss_G, mean_loss_D))

    
            writer.add_scalar("Valid-Epoch/loss_G_GAN", mean_loss_G_GAN_v, epoch)
            writer.add_scalar("Valid-Epoch/loss_G_L1", mean_loss_G_L1_v, epoch)
            writer.add_scalar("Valid-Epoch/loss_G", mean_loss_G_v, epoch)
            if wgan: writer.add_scalar("Valid-Epoch/loss_D_GAN", mean_loss_D_GAN_v, epoch)
            writer.add_scalar("Valid-Epoch/loss_D", mean_loss_D_v, epoch)
            
            # Test Generation
            if (epoch % 10 == 0):
                for i, batch in enumerate(val_dataloader):
                    if i == 1:
                        real = batch["mfsc_real"].to(device)
                        spec = batch["mfsc_synth"].to(device)
                        f0 = batch["f0"].to(device)
                        fake_source = torch.cat((spec,f0), 1)
                        test_generate(generator, real, fake_source, spec_transform=shift_scale(7, 0.1, inverse=True), title=("VALID epoch %d" % epoch), writer=writer, device=device, useNoise=useNoise)

        


if __name__ == "__main__":
    print("Test 24")
    generator = UnetGenerator(2, 1, 7) # 2 channels if no noise, 3 if noise
    discriminator = NLayerDiscriminator(3)

    dataset = SpecDataset(spec_transform=shift_scale(7, 0.1), f0_transform=shift_scale(-200, 0.005))
    train_dl, valid_dl = get_dataloaders(dataset, split=0.8)

    train(generator, discriminator, train_dl, valid_dl, 151, wgan=True)