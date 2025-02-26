# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import warnings

import torch
import torch.nn.functional as F
import torchvision
from torchmetrics.classification import BinaryAUROC
from transformers import BertForSequenceClassification, AdamW, get_scheduler


# It's okay if a batch only contains negative/positive samples
warnings.filterwarnings("ignore", message="No positive samples in targets, true positive value should be meaningless.")


class ToyNet(torch.nn.Module):
    def __init__(self, dim, gammas):
        super(ToyNet, self).__init__()
        # gammas is a list of three the first dimension determines how fast the
        # spurious feature is learned the second dimension determines how fast
        # the core feature is learned and the third dimension determines how
        # fast the noise features are learned
        self.register_buffer(
            "gammas", torch.tensor([gammas[:2] + gammas[2:] * (dim - 2)])
        )
        self.fc = torch.nn.Linear(dim, 1, bias=False)
        self.fc.weight.data = 0.01 / self.gammas * self.fc.weight.data

    def forward(self, x):
        return self.fc((x * self.gammas).float()).squeeze()


class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits


def get_bert_optim(network, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in network.named_parameters():
        if any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer


def get_sgd_optim(network, lr, weight_decay):
    return torch.optim.SGD(
        network.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9)


class ERM(torch.nn.Module):
    def __init__(self, hparams, dataloader):
        super().__init__()
        self.hparams = dict(hparams)
        dataset = dataloader.dataset
        self.n_batches = len(dataloader)
        self.data_type = dataset.data_type
        self.n_classes = len(set(dataset.y))
        self.n_groups = len(set([round(int(gg)) for gg in dataset.g]))
        self.n_examples = len(dataset)
        self.last_epoch = 0
        self.best_selec_val = 0
        self.init_model_(self.data_type)

    def init_model_(self, data_type, text_optim="sgd"):
        self.clip_grad = text_optim == "adamw"
        optimizers = {
            "adamw": get_bert_optim,
            "sgd": get_sgd_optim
        }

        if self.__class__.__name__ in { "TTLSA", "TTLSI", "Oracle", "BatchOracle" }:
            out_features = self.n_classes * self.n_groups
        else:
            out_features = self.n_classes

        if data_type in { "images", "mnist" }:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            self.network = torchvision.models.resnet.resnet50(weights=weights)
            self.network.fc = torch.nn.Linear(
                self.network.fc.in_features, out_features)

            self.optimizer = optimizers['sgd'](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay'])

            self.lr_scheduler = None
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")

        elif data_type == "text":
            self.network = BertWrapper(
                BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased', num_labels=out_features))
            self.network.zero_grad()
            self.optimizer = optimizers[text_optim](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay'])

            num_training_steps = self.hparams["num_epochs"] * self.n_batches
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps)
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")

        elif data_type == "embeddings":
            # TODO: generalize to embedding dimensions other than CXR Foundation
            self.network = torch.nn.Linear(1376, out_features)

            self.optimizer = optimizers['sgd'](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay'])

            self.lr_scheduler = None
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")

        elif data_type == "toy":
            gammas = (
                self.hparams['gamma_spu'],
                self.hparams['gamma_core'],
                self.hparams['gamma_noise'])

            self.network = ToyNet(self.hparams['dim_noise'] + 2, gammas)
            self.optimizer = optimizers['sgd'](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay'])
            self.lr_scheduler = None
            self.loss = lambda x, y:\
                torch.nn.BCEWithLogitsLoss(reduction="none")(x.squeeze(),
                                                             y.float())

        else:
            raise ValueError(f"Unknown data_type = '{data_type}'")

        self.cuda()

    def compute_loss_value_(self, i, x, y, g, epoch):
        # Normally y.dtype is torch.int64, and .long() is a no-op.
        # However, during imputation y is actually g, in which case we must call .long() since dtype.y is torch.float32.
        return self.loss(self.network(x), y.long()).mean()

    def update(self, i, x, y, g, epoch):
        x, y, g = x.cuda(), y.cuda(), g.cuda()
        loss_value = self.compute_loss_value_(i, x, y, g, epoch)

        if loss_value is not None:
            self.optimizer.zero_grad()
            loss_value.backward()

            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.data_type == "text":
                self.network.zero_grad()

            loss_value = loss_value.item()

        self.last_epoch = epoch
        return loss_value

    def predict(self, x, oracle_prior, batch_oracle_prior):
        return self.network(x)

    def accuracy(self, loader, predict_g=False):
        nb_groups = loader.dataset.nb_groups
        nb_labels = loader.dataset.nb_labels

        totals = torch.zeros(nb_groups * nb_labels, dtype=torch.long)
        for i, x, y, g in loader:
            g = torch.round(g).int()    # harden soft labels, assuming binary labels
            groups = nb_groups * y + g
            for gi in groups.unique():
                totals[gi] += (groups == gi).sum()
        oracle_prior = totals.float() / totals.sum()
        oracle_prior = oracle_prior.cuda()

        corrects = torch.zeros(nb_groups * nb_labels, dtype=torch.long)
        metric = BinaryAUROC()

        self.eval()
        with torch.inference_mode():
            for i, x, y, g in loader:
                g = torch.round(g).int()    # harden soft labels, assuming binary labels
                groups = nb_groups * y + g

                batch_oracle_prior = torch.bincount(groups, minlength=nb_groups * nb_labels)
                batch_oracle_prior = batch_oracle_prior.float() / batch_oracle_prior.sum()
                batch_oracle_prior = batch_oracle_prior.cuda()
                predictions = self.predict(x.cuda(), oracle_prior, batch_oracle_prior).cpu()

                labels = g if predict_g else y
                if predictions.squeeze().ndim == 1:
                    metric(predictions, labels)
                    predictions = (predictions > 0).eq(labels)
                else:
                    probability = torch.softmax(predictions, dim=-1)
                    metric(probability[:, 1], labels)
                    predictions = predictions.argmax(1).eq(labels)

                for gi in groups.unique():
                    corrects[gi] += predictions[(groups == gi)].sum()
        self.train()

        auc = metric.compute()
        acc = corrects.sum() / totals.sum()
        corrects, totals = corrects.tolist(), totals.tolist()
        group_accs = [c/t if t > 0 else float("nan") for c, t in zip(corrects, totals)]
        return auc.item(), acc.item(), corrects, totals, group_accs


    def load(self, fname):
        dicts = torch.load(fname)
        self.last_epoch = dicts["epoch"]
        self.load_state_dict(dicts["model"])
        self.optimizer.load_state_dict(dicts["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(dicts["scheduler"])

    def save(self, fname):
        lr_dict = None
        if self.lr_scheduler is not None:
            lr_dict = self.lr_scheduler.state_dict()
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": lr_dict,
                "epoch": self.last_epoch,
                "best_selec_val": self.best_selec_val,
            },
            fname,
        )


class GroupDRO(ERM):
    def __init__(self, hparams, dataset):
        super(GroupDRO, self).__init__(hparams, dataset)
        self.register_buffer("q", torch.ones(self.n_classes * self.n_groups).cuda())

    def groups_(self, y, g):
        idx_g, idx_b = [], []
        all_g = torch.round(y * self.n_groups + g).int()

        for g in all_g.unique():
            idx_g.append(g)
            idx_b.append(all_g == g)

        return zip(idx_g, idx_b)

    def compute_loss_value_(self, i, x, y, g, epoch):
        losses = self.loss(self.network(x), y)

        for idx_g, idx_b in self.groups_(y, g):
            self.q[idx_g] *= (self.hparams["eta"] * losses[idx_b].mean()).exp().item()

        self.q /= self.q.sum()

        loss_value = 0
        for idx_g, idx_b in self.groups_(y, g):
            loss_value += self.q[idx_g] * losses[idx_b].mean()

        return loss_value


class JTT(ERM):
    def __init__(self, hparams, dataset):
        super(JTT, self).__init__(hparams, dataset)
        self.register_buffer(
            "weights", torch.ones(self.n_examples, dtype=torch.long).cuda())

    def compute_loss_value_(self, i, x, y, g, epoch):
        if epoch == self.hparams["T"] + 1 and\
           self.last_epoch == self.hparams["T"]:
            self.init_model_(self.data_type, text_optim="adamw")

        predictions = self.network(x)

        if epoch != self.hparams["T"]:
            loss_value = self.loss(predictions, y).mean()
        else:
            self.eval()
            if predictions.squeeze().ndim == 1:
                wrong_predictions = (predictions > 0).ne(y)
            else:
                wrong_predictions = predictions.argmax(1).ne(y)

            self.weights[i] += wrong_predictions.detach() * (self.hparams["up"] - 1)
            self.train()
            loss_value = None

        return loss_value

    def load(self, fname):
        dicts = torch.load(fname)
        self.last_epoch = dicts["epoch"]

        if self.last_epoch > self.hparams["T"]:
            self.init_model_(self.data_type, text_optim="adamw")

        self.load_state_dict(dicts["model"])
        self.optimizer.load_state_dict(dicts["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(dicts["scheduler"])


class TTLSA(ERM):
    def __init__(self, hparams, dataloader):
        super().__init__(hparams, dataloader)
        self.register_buffer("source_prior", self._empirical_count(dataloader).cuda())

        self.register_parameter("T", torch.nn.Parameter(torch.ones(1).cuda()))
        self.register_parameter("b", torch.nn.Parameter(torch.zeros(self.n_classes * self.n_groups).cuda()))

        self.bcts_optimizer = torch.optim.SGD(
            [self.T, self.b],
            lr=1e-2,
            momentum=0.9)

    def _empirical_count(self, dataloader):
        dataset = dataloader.dataset
        y = torch.ByteTensor(dataset.y)
        g = torch.ByteTensor(dataset.g)
        m = y * self.n_groups + g
        bincount = torch.bincount(m, minlength=self.n_classes * self.n_groups)
        return bincount / torch.sum(bincount)

    def compute_loss_value_(self, i, x, y, g, epoch):
        y = F.one_hot(y, num_classes=self.n_classes)
        g = torch.vstack([1 - g, g]).T
        m = torch.reshape(y[:, :, None] * g[:, None, :], (-1, self.n_groups * self.n_classes))
        return self.loss(self.network(x) + torch.log(self.source_prior), m).mean()

    def calibrate(self, i, x, y, g, epoch):
        x, y, g = x.cuda(), y.cuda(), g.cuda()
        loss_value = self.compute_loss_value_calibrate_(i, x, y, g, epoch)

        if loss_value is not None:
            self.bcts_optimizer.zero_grad()
            loss_value.backward()

            self.bcts_optimizer.step()

            loss_value = loss_value.item()

        return loss_value

    def compute_loss_value_calibrate_(self, i, x, y, g, epoch):
        with torch.inference_mode():
            logits = self.network(x) + torch.log(self.source_prior)

        y = F.one_hot(y, num_classes=self.n_classes)
        g = torch.vstack([1 - g, g]).T
        m = torch.reshape(y[:, :, None] * g[:, None, :], (-1, self.n_groups * self.n_classes))
        return self.loss((logits - self.b) / torch.exp(self.T), m).mean()

    def predict(self, x, oracle_prior, batch_oracle_prior):
        logits = self.network(x)
        calibrated = (logits - self.b) / torch.exp(self.T)

        # calculate the MLE for target_prior
        prob = torch.softmax(calibrated, dim=-1)
        target_prior = self.source_prior
        last_objective = float("-inf")
        while True:
            # E step
            target_prob = prob * target_prior

            # calculate objective
            llk = torch.log(torch.sum(target_prob, dim=-1))
            objective = torch.sum(llk)
            if objective <= last_objective:
                break
            last_objective = objective

            # M step
            target_prob /= torch.sum(target_prob, dim=-1, keepdim=True)
            target_prior = torch.sum(target_prob, dim=0)
            target_prior /= torch.sum(target_prior, dim=-1, keepdim=True)

        target_prob = target_prob.reshape(-1, self.n_classes, self.n_groups)
        target_prob = target_prob.sum(dim=-1)

        return torch.log(target_prob)


class TTLSI(TTLSA):
    def predict(self, x, oracle_prior, batch_oracle_prior):
        logits = self.network(x)
        prob = torch.softmax(logits, dim=-1)
        prob = prob.reshape(-1, self.n_classes, self.n_groups)
        prob = prob.sum(dim=-1)
        return torch.log(prob)


class Oracle(TTLSA):
    def predict(self, x, oracle_prior, batch_oracle_prior):
        logits = self.network(x)
        calibrated = (logits - self.b) / torch.exp(self.T)
        prob = torch.softmax(calibrated, dim=-1)
        prob = prob * oracle_prior
        prob = prob.reshape(-1, self.n_classes, self.n_groups)
        prob = prob.sum(dim=-1)
        return torch.log(prob)


class BatchOracle(TTLSA):
    def predict(self, x, oracle_prior, batch_oracle_prior):
        logits = self.network(x)
        calibrated = (logits - self.b) / torch.exp(self.T)
        prob = torch.softmax(calibrated, dim=-1)
        prob = prob * batch_oracle_prior
        prob = prob.reshape(-1, self.n_classes, self.n_groups)
        prob = prob.sum(dim=-1)
        return torch.log(prob)
