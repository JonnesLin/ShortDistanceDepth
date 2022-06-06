"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import sys
sys.path.append(os.getcwd())

import torch as torch

from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model
from util.visualizer import simply_print
from tensorboardX import SummaryWriter
from tqdm import tqdm


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    # opt.serial_batches = True
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataloader = create_dataloader(opt, train=True)
    eval_dataloader = create_dataloader(opt, train=False)
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    writer = SummaryWriter(log_dir=opt.log_dir)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    # model.setup(opt)  # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iteration = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.epoch_count):
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        print('Epoch: %d' % epoch)
        total_loss = 0
        model.train()
        for i, (rgb_img, depth_img) in tqdm(enumerate(train_dataloader)):  # inner loop within one epoch
            epoch_iter += rgb_img.shape[0]
            total_iteration += 1
            model.set_input_train(rgb_img, depth_img)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            total_loss += model.get_current_losses()
        # Visualize training images
        model.write_visuals(writer, epoch, 'train')
        # Visualize training performance
        model.record_performance()
        model.write_performance(writer, epoch, 'train')
        writer.add_scalar('train/loss', total_loss/i, epoch)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            print('Epoch: %d' % epoch)
            for i, (rgb_img, depth_img) in tqdm(enumerate(eval_dataloader)):  # inner loop within one epoch
                total_iteration += 1
                model.set_input_train(rgb_img, depth_img)  # unpack data from dataset and apply preprocessing
                model.forward()
                model.calculate_loss()  # calculate loss functions
                total_loss += model.get_current_losses()

            # Visualize training images
            model.write_visuals(writer, epoch, 'eval')
            # Visualize training performance
            model.record_performance()
            model.write_performance(writer, epoch, 'eval')
            writer.add_scalar('eval/loss', total_loss / i, epoch)

        simply_print(total_loss/i, model.get_metrics())

