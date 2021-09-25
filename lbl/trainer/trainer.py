import numpy as np
import torch

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
        metrics = list()
        losses  = list()

        with torch.no_grad():
            for data, target in self.valid_data_loader:

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                loss = self.loss_function(output, target.argmax(dim=1)).item()
                losses.append(loss)

                out = torch.argmax(output, dim=1)
                ground_truths = torch.argmax(target, dim=1)

                a = torch.mean((out == ground_truths).type(torch.float32)).item()

                metrics.append(a)

        return np.mean(np.array(metrics)), np.mean(np.array(losses))


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
