# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np

import networks


class ERM(torch.nn.Module):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, model_type):
        super(ERM, self).__init__()
        self.featurizer = networks.Featurizer(model_type)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            2, #num classes
            False) #nonlinear

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1e-3, #default
            #weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.predict(x)

