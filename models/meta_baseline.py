import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query, optimal_prototype=None):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        x_shot_feature = x_shot
        x_query_feature = x_query
        if self.method == 'cos':
            if optimal_prototype is None:
                x_shot = x_shot.mean(dim=-2)
                x_shot_prototype = x_shot
                x_shot = F.normalize(x_shot, dim=-1)
                x_query = F.normalize(x_query, dim=-1)
                x_shot_prototype_norm = F.normalize(x_shot_prototype, dim=-1)
                metric = 'dot'
            else:
                x_shot = x_shot.mean(dim=-2)
                x_shot_prototype = x_shot
                x_shot = F.normalize(x_shot, dim=-1)
                x_query = F.normalize(x_query, dim=-1)
                optimal_prototype = F.normalize(optimal_prototype, dim=-1)
                metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            x_shot_prototype = x_shot
            metric = 'sqr'
        if optimal_prototype is None:
            logits = utils.compute_logits(
                    x_query, x_shot, metric=metric, temp=self.temp)

            optimial_logits = utils.compute_logits(
                    x_query, x_shot_prototype_norm, metric=metric, temp=self.temp)
        else:

            logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)

            optimial_logits = utils.compute_logits(
                x_query, optimal_prototype, metric=metric, temp=self.temp)

        return logits, x_shot_prototype, optimial_logits, x_shot_feature, x_query_feature

