import time
import sys

from typing import Union
from pathlib import Path
from datetime import datetime

import torch

import numpy as np


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: torch.nn,
                 optimizer: torch.optim,
                 epochs: int,
                 save_period: int,
                 savedir: str,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 device: str = None,
                 ):
        """
        Args:
            model (torch.nn.Module): The model to be trained
            loss_function (MultiLoss): The loss function or loss function class
            optimizer (torch.optim): torch.optim, i.e., the optimizer class
            config (dict): dict of configs
            lr_scheduler (torch.optim.lr_scheduler): pytorch lr_scheduler for manipulating the learning rate
            seed (int): integer seed to enforce non stochasticity,
            device (str): string of the device to be trained on, e.g., "cuda:0"
        """

        # Model to device
        self.device = torch.device(device)

        '''
        # setup GPU device if available, move model into configured device
        if device is None:
            self.device, device_ids = self.prepare_device(config['n_gpu'])
        else:
            self.device = torch.device(device)
            device_ids = list()
        '''

        self.model              = model.to(self.device)
        self.lr_scheduler       = lr_scheduler
        self.loss_function      = loss_function.to(self.device)
        self.optimizer          = optimizer

        self.epochs             = epochs
        self.save_period        = save_period
        self.start_epoch        = 1

        self.checkpoint_dir     = Path(savedir) / Path(datetime.today().strftime('%Y-%m-%d'))

        self.min_validation_loss = sys.float_info.max  # Minimum validation loss achieved, starting with the larges possible number


    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):

            epoch_start_time = time.time()

            loss = self._train_epoch(epoch)
            valid_acc, valid_loss = self._valid_epoch(epoch)

            epoch_end_time = time.time() - epoch_start_time

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print('Epoch/iteration {} with validation completed in {}, '\
                'run mean statistics:'.format(epoch, epoch_end_time))

            print('Current learning rate: {}'.format(self.lr_scheduler.get_last_lr()))

            # print logged informations to the screen
            print('Mean training loss: {}'.format(np.mean(loss)))
            print('Mean validation loss: {}'.format(np.mean(valid_loss)))
            print('Mean validation accuracy: {}'.format(valid_acc))


            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, best=False)

            #if val_loss < self.min_validation_loss:
            if 1 - valid_acc < self.min_validation_loss:
                self.min_validation_loss = 1 - valid_acc
                self.save_checkpoint(epoch, best=True)

            print('-----------------------------------')

        self.save_checkpoint(epoch, best=False)


    def save_checkpoint(self, epoch, best: bool = False):
        """
        Saving checkpoints at the given moment
        Args:
            epoch (int), the current epoch of the training
            bool (bool), save as best epoch so far, different naming convention
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            }

        if best:  # Save best case with different naming convention
            save_path = Path(self.checkpoint_dir) / Path('best_validation')
            filename = str(save_path / 'checkpoint-best.pth')
        else:
            save_path = Path(self.checkpoint_dir) / Path('epoch_' + str(epoch))
            filename = str(save_path / 'checkpoint-epoch{}.pth'.format(epoch))

        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

    def resume_checkpoint(self,
                          resume_model: Union[str, Path],
                          ):
        """
        Resume from saved checkpoints
        Args:
            resume_model (str, pathlib.Path): Checkpoint path, either absolute or relative
        """
        if not isinstance(resume_model, (str, Path)):
            print('resume_model is not str or Path object but of type {}, '
                                'aborting previous checkpoint loading'.format(type(resume_model)))
            return None

        if not Path(resume_model).is_file():
            print('resume_model object does not exist, ensure that {} is correct, '
                                'aborting previous checkpoint loading'.format(str(resume_model)))
            return None

        resume_model = str(resume_model)
        print("Loading checkpoint: {} ...".format(resume_model))


        checkpoint = torch.load(resume_model, map_location='cpu')

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.start_epoch = checkpoint['epoch'] + 1

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

        print('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

        self.checkpoint_dir = Path(resume_model).parent.parent  # Ensuring the same main folder after resuming

        # Fix this
        # for key, value in self.metric[self.metric.VALIDATION_KEY].items():
        #     loss = np.mean(np.array(value['loss']))
        #     self.min_validation_loss = min(self.min_validation_loss, loss)
