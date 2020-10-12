"""
train_process.py

Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import sys
import abc

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


_DEFAULT_CP_FREQ = 1000


class AbstractProcess(object):
    """Abstract training process"""

    __metaclass__  = abc.ABCMeta

    def __init__(self, optimizer, lr_scheduler,
                 check_point_freq=_DEFAULT_CP_FREQ, verbose=True):
        #, loss_buf_size=1000):
        """
        Args:
            optimizer : pytorch optimizer
            lr_scheduler : a implementation of LearningRateScheduler,
                           also checkes stopping condition
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.verbose = verbose
        self.check_point_freq = check_point_freq

    @abc.abstractmethod
    def process_batch():
        """
            load batch, apply network, and return loss

            Returns:
                Mean loss of the current batch
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def check_point(self):
        """
            this method is called every check_point_freq batches as well as
            at the end of training process.
            use this method to store the model maybe
        """
        raise NotImplementedError()

    def run(self):
        batch_counter = 0
        cont = True
        while cont:
            self.optimizer.zero_grad()
            loss = self.process_batch()
            loss.backward()
            self.optimizer.step()

            if self.verbose:
                print >> sys.stderr, 'Update %d, Loss: %.4f' % \
                                            (batch_counter, loss.item())
            batch_counter += 1

            cont, adjust, lr = self.lr_scheduler.update(loss.item())
            if adjust:
                self._adjust_lr(lr)

            if batch_counter % self.check_point_freq == 0:
                self.check_point()
        self.check_point()

    def _adjust_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if self.verbose:
                print >> sys.stderr, \
                         'Adjust learning rate : %g' % param_group['lr']


class LearningRateScheduler(object):

    __metaclass__  = abc.ABCMeta

    @abc.abstractmethod
    def update(self, loss):
        """
        update training status of this batch, and check if learning rate needs
        adjusted or stop condition is met

        Args:
            loss : mean loss of this batch

        Returns:
            cont   : False if stop condition is met
            adjust : True if learning rate needs adjustment
            lr     : new learning rate
        """
        raise NotImplementedError()

    def get_status(self):
        """
        get status as a string
        """
        return "get_status not implemented"


class DecreaseOnPlateau(LearningRateScheduler):

    def __init__(self, init_lr, num_decay, loss_buf_size=1000,
                 decay_factor=0.5, verbose=True):
        self.learning_rate = init_lr
        self.num_decay = num_decay
        self.loss_buf_size = loss_buf_size
        self.decay_factor = decay_factor
        self.verbose = verbose

        self.loss_buf_sum = 0.0
        self.prev_loss_avg = None
        self.batch_counter = 0

    def update(self, loss):
        self.loss_buf_sum += loss
        self.batch_counter = (self.batch_counter + 1) % self.loss_buf_size

        if self.batch_counter == 0:
            loss_avg = self.loss_buf_sum / self.loss_buf_size
            if self.verbose:
                print >> sys.stderr, 'Average of recent %d batches ' \
                         ': %.4f' % (self.loss_buf_size, loss_avg)
            self.loss_buf_sum = 0.0
            if (self.prev_loss_avg is not None
                    and loss_avg >= self.prev_loss_avg):
                self.prev_loss_avg = loss_avg
                self.num_decay -= 1
                if self.num_decay <= 0:
                    return False, False, self.learning_rate
                else:
                    self.learning_rate *= self.decay_factor
                    return True, True, self.learning_rate
            else:
                self.prev_loss_avg = loss_avg
        return True, False, self.learning_rate

    def get_status(self):
        if self.prev_loss_avg is not None:
            return "current loss: %.4f\nremaining decays: %d\n" \
                                    % (self.prev_loss_avg, self.num_decay)
        else:
            return "current loss: unknown\nremaining decays: %d\n" \
                                    % self.num_decay


class SimpleModelProcess(AbstractProcess):
    """
        training process with one simple model
    """

    def __init__(self, net, model_path, optimizer, lr_scheduler, gpu=True,
                 check_point_freq=_DEFAULT_CP_FREQ, verbose=True):
        super(SimpleModelProcess, self).__init__(
                optimizer, lr_scheduler, check_point_freq, verbose)
        self.net = net
        self.model_path = model_path
        self.gpu = gpu
        self._check_point_counter = 0

    def check_point(self):
        # Save model
        if self.gpu:
            self.net.cpu()
            self.net.save(self.model_path)
            self.net.cuda()
        else:
            self.net.save(self.model_path)
        self._store_status()
        self._check_point_counter += 1

    def _store_status(self):
        with open(self.model_path + '.stat', 'a') as f:
            print >> f, '#### Checkpoint #%d ####' % self._check_point_counter
            print >> f, self.lr_scheduler.get_status()


class SupervsiedProcess(SimpleModelProcess):
    """
        Standard supervised training process
    """

    def __init__(self, net, model_path, train_loader, criterion, optimizer,
                 lr_scheduler, gpu=True, check_point_freq=_DEFAULT_CP_FREQ,
                 verbose=True):
        super(SupervsiedProcess, self).__init__(
                net, model_path, optimizer, lr_scheduler, gpu,
                check_point_freq, verbose)
        # train loader
        self.train_loader = train_loader
        self.loader_iter = iter(self.train_loader)

        # loss
        if criterion is None:
            if gpu:
                criterion = nn.MSELoss().cuda()
            else:
                criterion = nn.MSELoss()
        self.criterion = criterion


    def process_batch(self):
        """
            load batch, apply network, and return loss

            Returns:
                Mean loss of the current batch
        """
        # load data
        try:
            x, y = next(self.loader_iter)
        except StopIteration:
            # end of epoch
            self.loader_iter = iter(self.train_loader)
            x, y = next(self.loader_iter)

        # Convert torch tensor to Variable
        if self.gpu:
            xv = Variable(x).cuda()
        else:
            xv = Variable(x)

        # forward
        ov = self.net(xv)

        # Convert torch tensor to Variable
        if self.gpu:
            yv = Variable(y).cuda()
        else:
            yv = Variable(y)

        loss = self.criterion(ov, yv)
        return loss


class AdaptationProcess(SimpleModelProcess):
    """
        Unsupervised adaptation process
    """

    def __init__(self, net, model_path, labeled_loader, adapt_loader,
                 adapt_func, criterion, optimizer, lr_scheduler,
                 adapt_criterion=None, gpu=True,
                 check_point_freq=_DEFAULT_CP_FREQ, verbose=True):
        super(AdaptationProcess, self).__init__(
                net, model_path, optimizer, lr_scheduler, gpu,
                check_point_freq, verbose)
        self.adapt_func = adapt_func

        # train loader
        self.labeled_loader = labeled_loader
        self.adapt_loader = adapt_loader
        self.labeled_loader_iter = iter(self.labeled_loader)
        self.adapt_loader_iter = iter(self.adapt_loader)

        # loss
        if criterion is None:
            if gpu:
                criterion = nn.MSELoss().cuda()
            else:
                criterion = nn.MSELoss()
        if adapt_criterion is None:
            adapt_criterion = criterion
        self.criterion = criterion
        self.adapt_criterion = adapt_criterion

    def process_batch(self):
        """
            load batch, apply network, and return loss

            Returns:
                Mean loss of the current batch
        """
        # load data
        #   - adapt data
        try:
            x, y = next(self.adapt_loader_iter)
        except StopIteration:
            # restart iterator
            self.adapt_loader_iter = iter(self.adapt_loader)
            x, y = next(self.adapt_loader_iter)
        #   - labeled data
        try:
            ox, oy = next(self.labeled_loader_iter)
        except StopIteration:
            # restart iterator
            self.labeled_loader_iter = iter(self.labeled_loader)
            ox, oy = next(self.labeled_loader_iter)

        # Convert torch tensor to Variable
        if self.gpu:
            xv = Variable(torch.cat([x, ox])).cuda()
        else:
            xv = Variable(torch.cat([x, ox]))

        # forward
        ov = self.net(xv)

        # apply unsupervised adaptation, replace ground truth y
        odata = ov.data[:len(x)]
        z = np.zeros(odata.size(), dtype='float32')
        for j in xrange(len(z)):
            z[j] = self.adapt_func(
                x[j].cpu().numpy(),
                odata[j].cpu().numpy(),
                y[j],
            )
        y = torch.from_numpy(z)

        # Convert torch tensor to Variable
        if self.gpu:
            yv = Variable(torch.cat([y, oy])).cuda()
        else:
            yv = Variable(torch.cat([y, oy]))

        aloss = self.adapt_criterion(ov[:len(y)], yv[:len(y)])
        oloss = self.criterion(ov[len(y):], yv[len(y):])
        loss = aloss + oloss

        if self.verbose:
            print >> sys.stderr, ('Loss detail : %.4f = %.4f + %.4f'
                                % (loss.item(), aloss.item(), oloss.item()))

        return loss


class AdaptDecomposedProcess(SimpleModelProcess):
    """
        Adaptation with augmentation by composition
    """

    def __init__(self, net, model_path, labeled_loader, augm_loader,
                 adapt_loader, adapt_func, augm_adapt_func, criterion,
                 optimizer, lr_scheduler, adapt_criterion=None, gpu=True,
                 check_point_freq=_DEFAULT_CP_FREQ, verbose=True,
                 fl_adapt_loader=None):
        super(AdaptDecomposedProcess, self).__init__(
                net, model_path, optimizer, lr_scheduler, gpu,
                check_point_freq, verbose)
        self.adapt_func = adapt_func
        self.augm_adapt_func = augm_adapt_func

        # train loader
        self.labeled_loader = labeled_loader
        self.fl_adapt_loader = fl_adapt_loader
        self.augm_loader = augm_loader
        self.adapt_loader = adapt_loader
        self.labeled_loader_iter = iter(self.labeled_loader)
        if fl_adapt_loader:
            self.fl_adapt_loader_iter = iter(self.fl_adapt_loader)
        self.augm_loader_iter = iter(self.augm_loader)
        self.adapt_loader_iter = iter(self.adapt_loader)

        # loss
        if criterion is None:
            if gpu:
                criterion = nn.MSELoss().cuda()
            else:
                criterion = nn.MSELoss()
        if adapt_criterion is None:
            adapt_criterion = criterion
        self.criterion = criterion
        self.adapt_criterion = adapt_criterion

    def process_batch(self):
        """
            load batch, apply network, and return loss

            Returns:
                Mean loss of the current batch
        """
        # load data
        #   - adapt data
        try:
            dx, dy = next(self.adapt_loader_iter)
        except StopIteration:
            # restart iterator
            self.adapt_loader_iter = iter(self.adapt_loader)
            dx, dy = next(self.adapt_loader_iter)

        #   - augmented data
        try:
            ax, ay = next(self.augm_loader_iter)
        except StopIteration:
            # restart iterator
            self.augm_loader_iter = iter(self.augm_loader)
            ax, ay = next(self.augm_loader_iter)

        #   - labeled data
        try:
            ox, oy = next(self.labeled_loader_iter)
        except StopIteration:
            # restart iterator
            self.labeled_loader_iter = iter(self.labeled_loader)
            ox, oy = next(self.labeled_loader_iter)

        #   - additional labeled adaptation data
        if self.fl_adapt_loader is not None:
            try:
                aox, aoy = next(self.fl_adapt_loader_iter)
            except StopIteration:
                # restart iterator
                self.fl_adapt_loader_iter = iter(self.fl_adapt_loader)
                aox, aoy = next(self.fl_adapt_loader_iter)
            ox = torch.cat([ox, aox])
            oy = torch.cat([oy, aoy])

        # for debug
        if self.verbose:
            if self.fl_adapt_loader is None:
                print >> sys.stderr, (
                    'Loaded batch: %d weakly labeled, %d augmented, %d labeled'
                    % (len(dy), len(ay), len(oy))
                )
            else:
                print >> sys.stderr, (
                    'Loaded batch: %d weakly labeled, %d augmented, %d labeled,'
                    ' %d labeled adpatation'
                    % (len(dy), len(ay), len(oy) - len(aoy), len(aoy))
                )

        # Convert torch tensor to Variable
        dxv = Variable(dx)
        axv = Variable(ax)
        oxv = Variable(ox)
        if self.gpu:
            dxv = dxv.cuda()
            axv = axv.cuda()
            oxv = oxv.cuda()

        # forward
        pv = self.net(torch.cat([dxv, axv[:, 0], oxv]))

        # apply network to single (decomposed)
        self.net.eval()
        pv1 = self.net(axv[:, 1])
        pv2 = self.net(axv[:, 2])
        self.net.train()

        # apply unsupervised adaptation, replace ground truth y
        # 1. adaptation data :
        pdata = pv.data[:len(dy)]
        z = np.zeros(pdata.size(), dtype='float32')
        for j in xrange(len(z)):
            z[j] = self.adapt_func(None, pdata[j].cpu().numpy(), dy[j])
        dy = torch.from_numpy(z)

        # 2. augmented data : use decomposed single source segments
        pdata1 = pv1.data
        pdata2 = pv2.data
        z = np.zeros(pdata1.size(), dtype='float32')
        for j in xrange(len(z)):
            z[j] = self.augm_adapt_func(pdata1[j].cpu().numpy(), ay[j, 1],
                                        pdata2[j].cpu().numpy(), ay[j, 2])
        ay = torch.from_numpy(z)

        # Convert torch tensor to Variable
        yv = Variable(torch.cat([dy, ay, oy]))
        if self.gpu:
            yv = yv.cuda()

        aloss = self.adapt_criterion(pv[:len(dy)+len(ay)], yv[:len(dy)+len(ay)])
        oloss = self.criterion(pv[len(dy)+len(ay):], yv[len(dy)+len(ay):])
        loss = aloss + oloss

        if self.verbose:
            print >> sys.stderr, ('Loss detail : %.4f = %.4f + %.4f'
                                % (loss.item(), aloss.item(), oloss.item()))

        return loss


# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

