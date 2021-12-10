import numpy as np
import torch
from sklearn.metrics import confusion_matrix

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

    def F_score(logit, label, threshold=0.5, beta=2):
        prob = torch.sigmoid(logit)
        prob = prob > threshold
        label = label > threshold

        TP = (prob & label).sum().float()
        TN = ((~prob) & (~label)).sum().float()
        FP = (prob & (~label)).sum().float()
        FN = ((~prob) & label).sum().float()

        accuracy = (TP+TN)/(TP+TN+FP+FN)
        precision = torch.mean(TP / (TP + FP + 1e-12))
        recall = torch.mean(TP / (TP + FN + 1e-12))
        F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)

        return accuracy, precision, recall, F2.mean(0)


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
        accuracy = list()    # metrics
        losses  = list()
        #preds = torch.tensor([])

        #y_true = [0]*4
        #y_pred = [0]*4
        n=4
        class_correct = [0]*n
        class_total   = [0]*n

        confusion_matrix = np.zeros((n,n)) #torch.zeros(4, 4)

        with torch.no_grad():
            for data, target in self.valid_data_loader:

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                loss = self.loss_function(output, target.argmax(dim=1)).item()
                losses.append(loss)

                out = torch.argmax(output, dim=1) # predicted
                ground_truths = torch.argmax(target, dim=1) # true class
                print(out)
                print(ground_truths)
                confusion_matrix[ground_truths][out] += 1
                print(confusion_matrix)

                _,pred = torch.max(output, 1)
                correct_tensor = ground_truths #pred.eq(target.data.view_as(pred))
                correct = np.squeeze(correct_tensor.numpy()) if self.device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())
                print(correct)
                for i in range(target.size(0)):
                    label = target.data[i]
                    print(label)
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

                    # Update confusion matrix
                    confusion_matrix[label][pred.data[i]] += 1
                    #confusion_matrix[ground_truths][out] += 1

                #accuracy, precision, recall, F1_score = F_score(output.squeeze(), labels.float())
                a = torch.mean((out == ground_truths).type(torch.float32)).item()
                accuracy.append(a)

                #for t, p in zip(target.view(-1), out.view(-1)):
                #    confusion_matrix[t.long(), p.long()] += 1

                #print(confusion_matrix)

                #y_pred = np.asarray(y_pred)
                #if y_pred.shape[1] > 1: #We have a classification problem, convert to labels
                #    y_pred = np.argmax(y_pred, axis=1)

                #print('pred: ', y_pred, 'true: ', y_true)

        print(accuracy)
        # valid_acc, valid_loss
        return np.mean(np.array(accuracy)), np.mean(np.array(losses)), confusion_matrix


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
