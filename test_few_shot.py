import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import datasets
import models
from models.meta_baseline import MetaBaseline
from models.transformer import Gpt
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from overfit import overfiting
from models import transformer
from diffusion.timestep_sampler import UniformSampler
from diffusion import create_diffusion
from diffusion import ema
from copy import deepcopy
import utils.sampling as sampling

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))
    if not args.sauc:
        n_way = 5
    else:
        n_way = 2
    n_shot, n_query = args.shot, 15
    n_batch = 200
    ep_per_batch = 1
    batch_sampler = CategoriesSampler(
            dataset.label, n_batch, n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=8, pin_memory=True)

    base_model = models.load(torch.load(config['load']))
    transformer_model = Gpt(
        prototype_sizes=512,
        predict_xstart=config["transformer"]["predict_xstart"],
        max_freq_log2=config["transformer"]['max_freq_log2'],
        num_frequencies=config["transformer"]['num_frequencies'],
        n_embd=config["transformer"]['n_embd'],
        encoder_depth=config["transformer"]['encoder_depth'],
        n_layer=config["transformer"]['n_layer'],
        n_head=config["transformer"]['n_head'],
        len_input=config["transformer"]['len_input'],
        attn_pdrop=config["transformer"]['dropout_prob'],
        resid_pdrop=config["transformer"]['dropout_prob'],
        embd_pdrop=config["transformer"]['dropout_prob'])
    transformer_model=transformer_model.cuda()
    state_dict = torch.load(config['load'])


    transformer_model.load_state_dict(state_dict['diff_model_sd'])


    diffusion = create_diffusion(learn_sigma=False, predict_xstart=config["transformer"]["predict_xstart"],
        noise_schedule='linear', steps=500)
    timestep_sampler = UniformSampler(diffusion)

    base_model.eval()
    transformer_model.eval()
    utils.log(' base_model num params: {}'.format(utils.compute_n_params(base_model)))
    utils.log('transformer_model num params: {}'.format(utils.compute_n_params(transformer_model)))
    # testing
    aves_keys = ['vl', 'va', 'vdl', 'vda']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst = []

    for epoch in range(1, test_epochs + 1):
        for data, _ in tqdm(loader, leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)
            label = fs.make_nk_label(n_way, n_query,
                                     ep_per_batch=ep_per_batch).cuda()

            predicted_prototype_list = []
            diffusion_loss_list = []
            original_prototype, original_shot_features, original_query_features = overfiting.train_overfitting(
                base_model, x_shot, x_query, label, n_way,
                lr=config["overfit_lr"],
                n_iteration=config["overfit_iterations"], Eval=True)

            predicted_prototype_eval_list = []
            for c in range(x_shot.shape[1]):
                predicted_prototype_eval = sampling.generalized_steps(transformer_model,
                                                                      original_shot_features[:, c, :],
                                                                      config['transformer']['time'])
                predicted_prototype_eval_list.append(predicted_prototype_eval)

                t, vlb_weights = timestep_sampler.sample(x_shot.shape[0], x_shot.device)
                diffusion_loss, predicted_prototype = diffusion.training_losses(transformer_model,
                                                                                predicted_prototype_eval,
                                                                                original_shot_features[:, c, :],
                                                                                t)
                predicted_prototype_list.append(torch.unsqueeze(predicted_prototype[:, 0, :], dim=1))
                diffusion_loss_list.append(diffusion_loss)
            predicted_prototypes = torch.cat(predicted_prototype_list, dim=1)

            with torch.no_grad():
                diffusion_losses = torch.cat(diffusion_loss_list)
                diffusion_losses_mean = torch.mean(diffusion_losses)
                ori_logits, _, diffusion_logits, _, _ = base_model(x_shot, x_query, predicted_prototypes)
                ori_logits = ori_logits.view(-1, n_way)
                diffusion_logits = diffusion_logits.view(-1, n_way)
                ce_loss = F.cross_entropy(diffusion_logits, label)
                ori_acc = utils.compute_acc(ori_logits, label)
                diffusion_acc = utils.compute_acc(diffusion_logits, label)

                final_loss = ce_loss + config["loss_coff"] * diffusion_losses_mean
                va_lst.append(diffusion_acc)

                aves['vl'].add(final_loss.item())
                aves['va'].add(ori_acc)
                aves['vdl'].add(diffusion_losses_mean.item())
                aves['vda'].add(diffusion_acc)


        print('test epoch {}: acc={:.2f}, ori_acc={:.2f} +- {:.2f} (%), loss={:.4f}'.format(
                epoch, aves['vda'].item() * 100, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--sauc', action='store_true', default=0)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)

