import numpy as np
import torch
import torch.nn.functional as F
import copy
import utils

def train_overfitting(bmodel, x_shot, x_query, label, n_train_way=5, lr=0.001, n_iteration=5):
    batch_prototype = []

    shot_shape = x_shot.shape[:-3]
    query_shape = x_query.shape[:-3]
    img_shape = x_shot.shape[-3:]
    x_query = x_query.view(-1, *img_shape)
    x_shot_ori = x_shot.view(-1, *img_shape)
    x_query = bmodel.encoder(x_query)
    x_query = x_query.view(*query_shape, -1)
    x_shot_ori = bmodel.encoder(x_shot_ori)
    x_shot_ori = x_shot_ori.view(*shot_shape, -1)
    x_query_feature = x_query.detach()
    x_query_norm = F.normalize(x_query_feature, dim=-1)


    for i in range(x_shot.shape[0]):
        base_model = copy.deepcopy(bmodel)
        param_weight = base_model.parameters()
        opt = torch.optim.Adam(
            param_weight, lr=lr)
        batch_shot = torch.unsqueeze(x_shot[i], dim=0)
        batch_label = label[i * int(label.shape[0] / x_shot.shape[0]):(i + 1) * int(label.shape[0] / x_shot.shape[0])]
        shot_shape = batch_shot[0].shape[:-3]

        for epoch in range(n_iteration):
            opt.zero_grad()
            batch_x_shot = batch_shot.view(-1, *img_shape)
            base_support_features = base_model.encoder(batch_x_shot)
            base_support_features = base_support_features.view(*shot_shape, -1)
            base_support_features = base_support_features.mean(dim=-2)
            x_shot_prototype_norm = F.normalize(base_support_features, dim=-1)

            predicted_labels = utils.compute_logits(
                x_query_norm[i], x_shot_prototype_norm, metric='dot')

            predicted_labels = predicted_labels.view(-1, n_train_way)
            loss = F.cross_entropy(predicted_labels, batch_label)
            # acc = utils.compute_acc(predicted_labels, batch_label)
            # print("epoch:" + str(epoch) + ", acc:" + str(acc) + ", loss:"+str(loss.item()))
            loss.backward()
            opt.step()

        batch_prototype.append(base_support_features)
        for param in base_model.parameters():
            param.detach()
    del base_model
    batch_prototype = torch.stack(batch_prototype).detach()

    return batch_prototype, x_shot_ori
