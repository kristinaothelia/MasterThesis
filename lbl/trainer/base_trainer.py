import time
import sys
import pandas as pd
import seaborn as sns
import termplotlib as tpl

from typing import Union
from pathlib import Path
from datetime import datetime

import torch

import numpy as np
import matplotlib.pyplot as plt


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: torch.nn,
                 optimizer: torch.optim,
                 epochs: int,
                 model_info: list(),
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

        self.model = model.to(self.device)
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function.to(self.device)
        self.optimizer = optimizer

        self.epochs = epochs
        self.model_info = model_info
        self.save_period = save_period
        self.start_epoch = 1

        self.checkpoint_dir = Path(savedir) / Path(datetime.today().strftime('%Y-%m-%d'))
        self.min_validation_loss = sys.float_info.max  # Minimum validation loss achieved, starting with the larges possible number


    def train(self):
        """
        Full training logic
        """
        t_loss = []
        v_loss = []
        v_acc  = []

        train_time = time.time()
        best_ep = 1
        best_acc = 0

        for epoch in range(self.start_epoch, self.epochs + 1):

            epoch_start_time = time.time()

            loss = self._train_epoch(epoch)
            valid_acc, valid_loss, confusion_matrix= self._valid_epoch(epoch)

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

            t_loss.append(np.mean(loss))
            v_loss.append(np.mean(valid_loss))
            v_acc.append(valid_acc)

            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, best=False)

            #if val_loss < self.min_validation_loss:
            if 1 - valid_acc < self.min_validation_loss:
                self.min_validation_loss = 1 - valid_acc
                self.save_checkpoint(epoch, best=True)
                best_ep = epoch
                best_acc = valid_acc
                best_conf_matrix = confusion_matrix
                print(best_conf_matrix)

            print('-----------------------------------')

        self.save_checkpoint(epoch, best=False)
        print("ep: ", best_ep, " acc: ", best_acc)

        train_end_time = time.time() - train_time
        print("Training time [h]: ", train_end_time/(60*60))

        log = open(self.checkpoint_dir / "log.txt", "w")
        log.write("Training time [h]: {}\n".format(train_end_time/(60*60)))
        log.write("The number of params in Million: {}\n".format(self.model_info[4]/1e6))
        log.write("Best epoch {} of {}. Validation acc: {}\n".format(best_ep, self.epochs, best_acc))
        log.write("Batch size (train): {}\n".format(self.model_info[0]))
        log.write("Other model info: lr:{}, step:{}, gamma:{}".format(self.model_info[1], self.model_info[2], self.model_info[3]))
        log.close()

        print(best_conf_matrix)

        plt.figure()
        class_names = ['no aurora', 'arc', 'diffuse', 'discrete']
        df_cm = pd.DataFrame(best_conf_matrix, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.show(block=True)
        plt.tight_layout()
        plt.savefig(str(self.checkpoint_dir) + "/CM.png")

        # Normalized
        N_cm = confusion_matrix/confusion_matrix.sum(axis=1)#[:, np.newaxis] #.astype('float')
        print(N_cm)

        plt.figure() # figsize=(15,10)
        df_cm = pd.DataFrame(N_cm, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.show(block=True)
        plt.tight_layout()
        plt.savefig(str(self.checkpoint_dir) + "/CM_normalized.png")


        if epoch == self.epochs:
            plt.figure()
            ep = np.linspace(self.start_epoch, self.epochs, self.epochs) # NB! change
            plt.title("Loss vs Accuracy. (best v.acc: {:.4f})".format(best_acc))
            plt.plot(ep, t_loss, label="Training loss")
            plt.plot(ep, v_loss, label="Validation loss")
            plt.plot(ep, v_acc, label="Validation accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(str(self.checkpoint_dir) + "/acc_vs_loss.png")

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
            #'optimizer': self.optimizer.state_dict(),
            #'scheduler': self.lr_scheduler.state_dict(),
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
