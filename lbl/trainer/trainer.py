import numpy as np
import torch
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
                    'Epoch',
                    epoch,
                    self._progress(batch_idx),
                    loss))

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

        n=4
        class_correct = [0]*n
        class_total   = [0]*n

        confusion_matrix = np.zeros((n,n))

        with torch.no_grad():
            for data, target in self.valid_data_loader:

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                loss = self.loss_function(output, target.argmax(dim=1)).item()
                losses.append(loss)

                out = torch.argmax(output, dim=1) # predicted
                ground_truths = torch.argmax(target, dim=1) # true class

                confusion_matrix[ground_truths][out] += 1

                #accuracy, precision, recall, F1_score = F_score(output.squeeze(), labels.float())
                a = torch.mean((out == ground_truths).type(torch.float32)).item()
                accuracy.append(a)

        report = sklearn.metrics.classification_report(ground_truths, out, target_names=['no a','arc','diff','disc'])
        print(report)
        
        #print('ground truths (true)')
        print(torch.shape(target))
        print(torch.shape(ground_truths))
        print(torch.shape(out))
        print(len(target))
        print(len(ground_truths))
        #print('out (pred)')
        print(len(out))

        #print('f1, average=None: ',f1_score(ground_truths, out, average=None))
        # The class F-1 scores are averaged by using the number of instances in a class as weights
        #print('f1, a=weighted:   ', f1_score(ground_truths, out, average='weighted'))
        #print('acc (calc):       ', valid_acc)
        #print('acc (sklearn):    ', accuracy_score(ground_truths, out))
        #print('acc (sklearn, b): ', balanced_accuracy_score(ground_truths, out))

        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        f1 = f1_score(ground_truths, out, average=None) #The best value is 1 and the worst value is 0
        f1_w = f1_score(ground_truths, out, average='weighted')
        acc_sk =accuracy_score(ground_truths, out)

        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
        acc_sk_b = balanced_accuracy_score(ground_truths, out) #The best value is 1 and the worst value is 0 when adjusted=False
        CM_sk = sklearn.metrics.confusion_matrix(ground_truths, out)

        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
        recall = sklearn.metrics.recall_score(ground_truths, out, average='weighted') #The best value is 1 and the worst value is 0

        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
        precision = sklearn.metrics.precision_score(ground_truths, out, average='weighted') #The best value is 1 and the worst value is 0

        return np.mean(np.array(accuracy)), np.mean(np.array(losses)), confusion_matrix, CM_sk, acc_sk, acc_sk_w, f1, f1_w, recall, precision


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
