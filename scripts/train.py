
#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import json
import os
import pathlib
import uuid

import dgl.nn
import torch
import tqdm.auto as tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gnngls import models, datasets


def train(model, data_loader, target, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for batch_i, batch in enumerate(train_loader):
        batch = batch.to(device)
        x = batch.ndata['features']
        y = batch.ndata[target]

        optimizer.zero_grad()
        y_pred = model(batch, x)
        loss = criterion(y_pred, y.type_as(y_pred))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= (batch_i + 1)
    return epoch_loss


def test(model, data_loader, target, criterion, device):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        for batch_i, batch in enumerate(data_loader):
            batch = batch.to(device)
            x = batch.ndata['features']
            y = batch.ndata[target]

            y_pred = model(batch, x)
            loss = criterion(y_pred, y.type_as(y_pred))

            epoch_loss += loss.item()

        epoch_loss /= (batch_i + 1)
        return epoch_loss


def save(model, optimizer, epoch, train_loss, val_loss, save_path):
    torch.save({