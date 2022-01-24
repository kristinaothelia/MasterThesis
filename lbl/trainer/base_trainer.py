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
        v_acc_w = []
        v_f1_w = []

        train_time = time.time()
        best_ep = 1
        best_acc = 0

        for epoch in range(self.start_epoch, self.epochs + 1):

            epoch_start_time = time.time()

            loss = self._train_epoch(epoch)
            #valid_acc, valid_loss, confusion_matrix, CM_sk, acc_sk, acc_sk_w, f1, f1_w, recall, precision, report = self._valid_epoch(epoch)
            valid_acc, valid_loss, CM_sk, acc_sk, acc_sk_w, f1, f1_w, recall, precision, report = self._valid_epoch(epoch)

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
            v_acc_w.append(acc_sk_w)
            v_f1_w.append(f1_w)

            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, best=False)

            #if val_loss < self.min_validation_loss:
            if 1 - valid_acc < self.min_validation_loss:
                self.min_validation_loss = 1 - valid_acc
                self.save_checkpoint(epoch, best=True)
                best_ep = epoch
                best_acc = valid_acc
                #best_conf_matrix = confusion_matrix
                L_v = np.mean(valid_loss)
                L_t = np.mean(loss)

                best_acc_sk = acc_sk
                best_acc_sk_w = acc_sk_w
                best_f1 = f1
                best_f1_w = f1_w
                best_CM_sk = CM_sk
                best_recall = recall
                best_precission = precision
                best_report = report

            print('-----------------------------------')

        self.save_checkpoint(epoch, best=False)

        print(best_report)
        print("ep: ", best_ep, " acc: ", best_acc)
        print('acc, sk:    ', best_acc_sk)
        print('acc, sk, w: ', best_acc_sk_w)
        print('f1, w:      ', best_f1_w)

        cm_ = best_CM_sk
        print(cm_.astype('float') / cm_.sum(axis=1)[:, np.newaxis])

        train_end_time = time.time() - train_time
        print("Training time [h]: ", train_end_time/(60*60))

        log = open(self.checkpoint_dir / "log.txt", "w")
        log.write("Model {}\n".format(self.model_info[-2]))
        log.write("Class weights {}\n".format(self.model_info[-1]))
        log.write("Training time [h]: {}\n".format(train_end_time/(60*60)))
        log.write("The number of params in Million: {}\n".format(self.model_info[4]))
        log.write("Best epoch {} of {}. Validation acc: {}\n".format(best_ep, self.epochs, best_acc))
        log.write("Training loss: {}. Validation loss: {}\n".format(L_t, L_v))
        log.write("Batch size (train): {}\n".format(self.model_info[0]))
        log.write("Other model info: lr:{}, step:{}, gamma:{}\n".format(self.model_info[1], self.model_info[2], self.model_info[3]))
        log.write("V.acc all 200 epochs: {} pm {}\n\n".format(np.mean(v_acc), np.std(v_acc)))
        log.write("recall: {}\n".format(best_recall))
        log.write("precision: {}\n".format(best_precission))
        log.write("f1 score (all classes): {}\n".format(best_f1))
        log.write("f1 score (w): {}. acc (w): {}\n\n".format(best_f1_w, best_acc_sk_w))
        log.write(best_report)

        log.close()

        #print(best_conf_matrix)


        # Normalized
        N_cm = best_CM_sk/best_CM_sk.sum(axis=1)[:, np.newaxis] #.astype('float')
        #N_cm = best_conf_matrix/best_conf_matrix.sum(axis=1)[:, np.newaxis] #.astype('float')
        class_names = [r'no aurora', r'arc', r'diffuse', r'discrete']
        #class_names = [r'no aurora', r'aurora']

        plt.figure() # figsize=(15,10)
        df_cm = pd.DataFrame(N_cm, index=class_names, columns=class_names).astype(float)
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=12)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=12)
        plt.ylabel(r'Observed class',fontsize=13) # True label
        plt.xlabel(r'Predicted class',fontsize=13)
        plt.title(r'Norm. confusion matrix for EfficientNet model B{}'.format(self.model_info[-2])+'\n'+r'Validation accuracy: {:.2f}'.format(best_acc),fontsize=14)
        #plt.show(block=True)
        plt.tight_layout()
        plt.savefig(str(self.checkpoint_dir) + "/CM_normalized.png")


        if epoch == self.epochs:
            ep = np.linspace(self.start_epoch, self.epochs, self.epochs)

            plt.figure()
            plt.title("Loss and accuracy for EfficientNet model B{}".format(self.model_info[-2])+'\n'+r'Save point validation accuracy: {:.2f}'.format(best_acc),fontsize=14)
            plt.plot(ep, t_loss, label="Training loss")
            plt.plot(ep, v_loss, label="Validation loss")
            plt.plot(ep, v_acc, label="Validation accuracy")
            plt.plot(best_ep, best_acc, 'r*', label="save point")
            plt.axhline(y=1, ls='--', color='lightgrey')
            plt.xlabel("Epochs",fontsize=13)
            plt.ylabel("Cross-Entropy loss, Accuracy",fontsize=13)
            plt.ylim(0, 1.5)
            plt.legend()
            plt.savefig(str(self.checkpoint_dir) + "/acc_vs_loss.png")

            plt.figure()
            plt.title("Model B{}, w accuracy and f1 score".format(self.model_info[-2]),fontsize=14)
            plt.plot(ep, v_acc, label="val. accuracy")
            plt.plot(ep, v_f1_w, label="Weighted f1 score")
            plt.plot(best_ep, best_acc, 'r*', label="save point")
            plt.axhline(y=1, ls='--', color='lightgrey')
            plt.xlabel("Epochs",fontsize=13)
            plt.ylabel("Accuracy, f1 score",fontsize=13)
            plt.legend()
            #plt.savefig(str(self.checkpoint_dir) + "/f1_acc.png")

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
