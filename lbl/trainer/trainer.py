import numpy as np
import torch
import sklearn as sk
#import torchvision.transforms.functional as TF

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: torch.nn,
                 optimizer: torch.optim,
                 data_loader: torch.utils.data.dataloader,
                 valid_data_loader: torch.utils.data.dataloader,
                 lr_scheduler: torch.optim.lr_scheduler,
                 epochs:        int,
                 model_info: list(),
                 save_period:   int,
                 savedir:       str,
                 device:        str = None,
                 log_step:      int = None,
                 ):

        super().__init__(model          = model,
                         loss_function  = loss_function,
                         optimizer      = optimizer,
                         lr_scheduler   = lr_scheduler,
                         device         = device,
                         epochs         = epochs,
                         model_info     = model_info,
                         save_period    = save_period,
                         savedir        = savedir,
                         )

        self.data_loader        = data_loader
        self.valid_data_loader  = valid_data_loader
        self.batch_size         = data_loader.batch_size
        self.len_epoch          = len(data_loader)*self.batch_size
        self.log_step           = int(self.len_epoch/(4)) if not isinstance(log_step, int) else log_step


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        losses = list()

        for batch_idx, (data, target) in enumerate(self.data_loader):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output  = self.model(data)
            loss    = self.loss_function(output, target.argmax(dim=1))

            loss.backward()
            self.optimizer.step()

            loss = loss.item()  # Detach loss from comp graph and moves it to the cpu
            losses.append(loss)

            if batch_idx % self.log_step == 0:
                print('Train {}: {} {} Loss: {:.6f}'.format(
                    'Epoch', epoch, self._progress(batch_idx), loss))

        return np.mean(np.array(losses))

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        accuracy = list()
        losses  = list()
        y_pred = list()
        y_true = list()

        wrong = list()

        with torch.no_grad():
            for data, target in self.valid_data_loader:

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                loss = self.loss_function(output, target.argmax(dim=1)).item()
                losses.append(loss)

                out = torch.argmax(output, dim=1) # predicted
                ground_truths = torch.argmax(target, dim=1) # true class

                a = torch.mean((out == ground_truths).type(torch.float32)).item()
                accuracy.append(a)

                # Update y_pred and y_true
                y_pred.extend(prediction.item() for prediction in out)
                y_true.extend(true.item() for true in ground_truths)

        def metrics(y_true, y_pred):
            report = sk.metrics.classification_report(y_true, y_pred, target_names=['no a','arc','diff','disc'])
            #report = sk.metrics.classification_report(y_true, y_pred, target_names=['no a','aurora'])
            #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
            f1 = f1_score(y_true, y_pred, average=None) #The best value is 1 and the worst value is 0
            f1_w = f1_score(y_true, y_pred, average='weighted')
            #accuracy =accuracy_score(y_true, y_pred, out) # sample_weightarray-like of shape (n_samples,) (same n as y_pred/true)
            accuracy =accuracy_score(y_true, y_pred)

            #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
            accuracy_w = balanced_accuracy_score(y_true, y_pred) #The best value is 1 and the worst value is 0 when adjusted=False
            CM_sk = sk.metrics.confusion_matrix(y_true, y_pred, normalize='true')

            #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
            recall = sk.metrics.recall_score(y_true, y_pred, average='weighted') #The best value is 1 and the worst value is 0

            #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
            precision = sk.metrics.precision_score(y_true, y_pred, average='weighted') #The best value is 1 and the worst value is 0

            return CM_sk, accuracy, accuracy_w, f1, f1_w, recall, precision, report

        # Metrics made with sklearn
        CM_sk, acc_sk, acc_sk_w, f1, f1_w, recall, precision, report = metrics(y_true, y_pred)

        valid_acc = np.mean(np.array(accuracy))
        valid_loss = np.mean(np.array(losses))

        return valid_acc, valid_loss, CM_sk, acc_sk, acc_sk_w, f1, f1_w, recall, precision, report


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'

        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        elif hasattr(self.data_loader, 'batch_size'):
            current = batch_idx * self.data_loader.batch_size
            total = self.len_epoch
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
