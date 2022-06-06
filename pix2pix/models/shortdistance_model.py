from abc import ABC

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base_model import BaseModel
from . import networks
from midas.models.midas_net import MidasNet
import torchvision.utils as vutils
from ..util.visualizer import colormap
from ..util.metrics import compute_metrics
import numpy as np
from .loss_functions import ScaleAndShiftInvariantLoss


class ShortDistanceModel(BaseModel, ABC):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(input_nc=2, output_nc=1, norm='none', netG='unet_1024', dataset_mode='depthmerge')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla', )
            parser.add_argument('--lambda_L1', type=float, default=1000, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # define networks
        self.model = MidasNet(path=None).to(self.device)

        # define criterion
        self.criterion = ScaleAndShiftInvariantLoss().to(self.device)  # torch.nn.MSELoss().to(self.device)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(opt.beta1, 0.999))

        self.rgb_img = None
        self.depth_map_img = None
        self.pred_depth_map_img = None
        self.loss = None
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=opt.n_epochs, eta_min=1e-6)
        self.total_loss = 0
        self.total_metrics = [0] * 7
        self.total_times = 0

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def set_input_train(self, rgb_img, depth_map_img):
        self.rgb_img = rgb_img.to(self.device)
        self.depth_map_img = depth_map_img.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred_depth_map_img = self.model(self.rgb_img)

    def calculate_loss(self):
        mask = torch.ones_like(self.rgb_img[:,0,:,:]).to(self.rgb_img.device)
        self.loss = self.criterion(self.pred_depth_map_img, self.depth_map_img, mask)

    def update_model(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def optimize_parameters(self):
        # compute a depth map
        self.forward()
        self.calculate_loss()
        self.update_model()

    def get_current_losses(self):
        return self.loss

    def get_metrics(self):
        return np.array(self.total_metrics) / self.total_times

    def get_current_visuals(self):
        rgb_img_vis = vutils.make_grid(self.rgb_img * 0.5 + 0.5)
        depth_img_vis = vutils.make_grid(colormap(self.depth_map_img.unsqueeze(1), 'gray'))
        pred_depth_map_img_vis = vutils.make_grid(colormap(self.pred_depth_map_img.unsqueeze(1), 'gray'))
        return rgb_img_vis, depth_img_vis, pred_depth_map_img_vis

    def write_visuals(self, writer, epoch, name):
        rgb_img_vis, depth_img_vis, pred_depth_map_img_vis = self.get_current_visuals()
        writer.add_image('%s/inputs' % name, rgb_img_vis, epoch)
        writer.add_image('%s/gt' % name, depth_img_vis, epoch)
        writer.add_image('%s/pred_depth' % name, pred_depth_map_img_vis, epoch)

    def record_performance(self):
        metrics = compute_metrics(self.pred_depth_map_img, self.depth_map_img)
        for idx, item in enumerate(self.total_metrics):
            self.total_metrics[idx] += metrics[idx]
        self.total_times += 1

    def write_performance(self, writer, epoch, name):
        log = np.array(self.total_metrics) / self.total_times
        writer.add_scalar('%s/rmse' % name, log[0], epoch)
        writer.add_scalar('%s/log_rms' % name, log[1], epoch)
        writer.add_scalar('%s/absrel' % name, log[2], epoch)
        writer.add_scalar('%s/sqrel' % name, log[3], epoch)
        writer.add_scalar('%s/a1' % name, log[4], epoch)
        writer.add_scalar('%s/a2' % name, log[5], epoch)
        writer.add_scalar('%s/a3' % name, log[6], epoch)
        self.total_times = 0
        self.total_metrics = [0] * 7
