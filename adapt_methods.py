import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
import numpy as np

from sam import SAM

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

class SAR(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model, self.optimizer = self.configure_model_optimizer(model)
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=True):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = self.forward_and_adapt_sar(x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        return outputs
    
    def update_ema(self, ema, new_data):
        if ema is None:
            return new_data
        else:
            with torch.no_grad():
                return 0.9 * ema + (1 - 0.9) * new_data
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_sar(self, x, model, optimizer, margin, reset_constant, ema):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        optimizer.zero_grad()
        # forward
        outputs = model(x)
        # adapt
        # filtering reliable samples/gradients for further adaptation; first time forward
        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < margin)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        entropys2 = softmax_entropy(model(x))
        entropys2 = entropys2[filter_ids_1]  # second time forward  
        loss_second_value = entropys2.clone().detach().mean(0)
        filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            ema = self.update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()
        optimizer.second_step(zero_grad=True)

        # perform model recovery
        reset_flag = False
        if ema is not None:
            if ema < 0.2:
                print("ema < 0.2, now reset the model")
                reset_flag = True

        return outputs, ema, reset_flag
    
    def collect_params(self, model):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names
    
    def configure_model_optimizer(self, algorithm):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = self.configure_model(adapted_algorithm.featurizer)
        params, param_names = self.collect_params(adapted_algorithm.featurizer)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=1e-3, momentum=0.9)
        # adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer
    
    def configure_model(self, model):
        """Configure model for use with SAR."""
        # train mode, because SAR optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what SAR updates
        model.requires_grad_(False)
        # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return model

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

class T3A(nn.Module):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, algorithm, num_classes=2, filter_K=5):
        super().__init__()
        self.featurizer = algorithm.featurizer
        self.classifier = algorithm.classifier

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = filter_K
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=True):
        cached_loader = False
        if not cached_loader:
            z = self.featurizer(x)
        else:
            z = x
        if adapt:
            # online adaptation
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to('cuda')

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to('cuda')
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=True):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data


class PseudoLabel(nn.Module):
    def __init__(self, algorithm, alpha=1, beta=0.9, gamma=1):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__()
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=alpha)
        self.beta = beta
        self.steps = gamma
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=True):
        cached_loader = False
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if cached_loader:
                    outputs = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                    self.model.featurizer.train()
        else:
            if cached_loader:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        
        loss = F.cross_entropy(outputs[flag], y_prime[flag])
        loss.backward()
        optimizer.step()
        return outputs

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.parameters(), 
            lr=1e-3  * alpha,
            #weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer

    def predict(self, x, adapt=True):
        return self(x, adapt)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
class SHOT(nn.Module):
    """
    "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation"
    """
    def __init__(self,algorithm, alpha=1,beta=0.9,theta=0.1,gamma=1):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        theta (float) : clf coefficient
        gamma (int) : number of updates
        """
        super().__init__()
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=alpha)
        self.beta = beta
        self.theta = theta
        self.steps = gamma
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=True):
        cached_loader = False
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if cached_loader:
                    outputs = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                    self.model.featurizer.train()
        else:
            if cached_loader:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        self.optimizer.zero_grad()
        outputs = model(x)
        
        loss = self.loss(outputs)
        loss.backward()
        self.optimizer.step()
        return outputs
    
    def loss(self, outputs):
        # (1) entropy
        ent_loss = softmax_entropy(outputs).mean(0)

        # (2) diversity
        softmax_out = F.softmax(outputs, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

        # (3) pseudo label
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        clf_loss = F.cross_entropy(outputs[flag], y_prime[flag])

        loss = ent_loss + self.theta * clf_loss
        return loss

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.featurizer.parameters(), 
            # adapted_algorithm.classifier.parameters(), 
            lr=1e-3  * alpha,
            #weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
class PLClf(PseudoLabel):
    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier.parameters(), 
            lr=1e-3 * alpha,
            #weight_decay=algorithm.hparams['weight_decay']
        )
        return adapted_algorithm, optimizer

    def predict(self, x, adapt=True):
        return self(x, adapt)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)