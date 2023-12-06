import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
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
def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
            config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder'] + '-' + str(
            config['overfit_lr']) + '-' + \
                  str(config['overfit_iterations'])
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
        train_dataset[0][0].shape, len(train_dataset),
        train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    train_sampler = CategoriesSampler(
        train_dataset.label, config['train_batches'],
        n_train_way, n_train_shot + n_query,
        ep_per_batch=ep_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)



    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
        val_dataset[0][0].shape, len(val_dataset),
        val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
        val_dataset.label, 100,
        n_way, n_shot + n_query,
        ep_per_batch=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    ########

    #### Model and optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)
    if not os.path.exists(args.meta_acc_logdir + "_" + str(config['n_shot'])):
        os.mkdir(args.meta_acc_logdir + "_" + str(config['n_shot']))

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    transformer_model = transformer.Gpt(
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
    transformer_model = transformer_model.cuda()
    diffusion = create_diffusion(learn_sigma=False, predict_xstart=config["transformer"]["predict_xstart"],
        noise_schedule='linear', steps=config["transformer"]["time"])
    timestep_sampler = UniformSampler(diffusion)

    ema_helper = ema.EMAHelper(mu=0.9999)
    ema_helper.register(transformer_model)

    optimizer_model, lr_scheduler_model = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], **config['optimizer_args'])
    optimizer_diffusion, lr_scheduler_diffusion = utils.make_optimizer(
        transformer_model.parameters(),
        config['diffusion_optimizer'], **config['diffusion_optimizer_args'])
    ########

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'tda', 'tvl', 'tva', 'vl', 'va', 'toa', 'tdl', 'tcel', 'to', 'tvda', 'vda', 'tvdl', 'vdl']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []
    best_test_acc = 0.

    best_test_epoch = 0.

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        transformer_model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model)
        writer.add_scalar('lr', optimizer_model.param_groups[0]['lr'], epoch)

        np.random.seed(epoch)
        count = []

        for data, _ in tqdm(train_loader, desc='train', leave=False):
            count.append(1)
            x_shot, x_query = fs.split_shot_query(
                data.cuda(), n_train_way, n_train_shot, n_query,
                ep_per_batch=ep_per_batch)
            label = fs.make_nk_label(n_train_way, n_query,
                                     ep_per_batch=ep_per_batch).cuda()
            overfited_prototype, original_shot_features = overfiting.train_overfitting(model, x_shot, x_query, label, n_train_way,
                                                               lr=config["overfit_lr"],
                                                               n_iteration=config["overfit_iterations"])



            predicted_prototype_list = []
            diffusion_loss_list = []
            for c in range(x_shot.shape[1]):
                t, vlb_weights = timestep_sampler.sample(x_shot.shape[0], x_shot.device)
                diffusion_loss, predicted_prototype_norm = diffusion.training_losses(transformer_model, overfited_prototype[:, c, :], original_shot_features[:, c, :], t)
                predicted_prototype = predicted_prototype_norm[:, 0, :] + torch.mean(original_shot_features[:, c, :, :], dim=1)
                predicted_prototype_list.append(torch.unsqueeze(predicted_prototype, dim=1))

                diffusion_loss_list.append(diffusion_loss)
            predicted_prototypes = torch.cat(predicted_prototype_list, dim=1)

            diffusion_losses = torch.cat(diffusion_loss_list)
            diffusion_losses_mean = torch.mean(diffusion_losses)

            ori_logits, _, diffusion_logits, _, _ = model(x_shot, x_query, predicted_prototypes)
            ori_logits = ori_logits.view(-1, n_train_way)
            diffusion_logits = diffusion_logits.view(-1, n_train_way)

            ce_loss = F.cross_entropy(diffusion_logits, label)
            ori_acc = utils.compute_acc(ori_logits, label)
            diffusion_acc = utils.compute_acc(diffusion_logits, label)

            final_loss = ce_loss + config["loss_coff"] * diffusion_losses_mean
            optimizer_model.zero_grad()
            optimizer_diffusion.zero_grad()
            final_loss.backward()
            optimizer_model.step()
            optimizer_diffusion.step()
            ema_helper.update(transformer_model)
            aves['tcel'].add(ce_loss.item())
            aves['tdl'].add(diffusion_losses_mean.item())
            aves['tl'].add(final_loss.item())
            aves['tda'].add(diffusion_acc)
            aves['toa'].add(ori_acc)
            if len(count) % 100 == 0:
                train_log = "epoch: %d, iteration: %d, diffusion_loss: %4f, ce_loss: %4f, final_loss: %4f, ori_acc: %4f, final_acc: %4f"%(
                            epoch, len(count), diffusion_losses_mean.item(), ce_loss.item(), final_loss.item(), ori_acc, diffusion_acc)
                log_file = open(args.meta_acc_logdir + "_" + str(config['n_shot']) + '/' + 'log.txt', "a")
                utils.print_and_log(log_file, train_log)
            # eval
        model.eval()
        transformer_model.eval()
        for name, loader, name_l, name_a, name_dl, name_da in [
            ('val', val_loader, 'vl', 'va', 'vdl', 'vda')]:
            if (config.get('tval_dataset') is None) and name == 'tval':
                continue
            test_acc = []
            test_ori_acc = []
            np.random.seed(0)
            for data, _ in tqdm(loader, desc=name, leave=False):
                x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=4)
                label = fs.make_nk_label(n_way, n_query,
                                         ep_per_batch=4).cuda()
                predicted_prototype_list = []
                diffusion_loss_list = []


                shot_shape = x_shot.shape[:-3]
                img_shape = x_shot.shape[-3:]
                x_shot_ori = x_shot.view(-1, *img_shape)
                x_shot_ori = model.encoder(x_shot_ori)
                original_shot_features = x_shot_ori.view(*shot_shape, -1)

                predicted_prototype_eval_list = []
                for c in range(x_shot.shape[1]):
                    predicted_prototype_eval = sampling.generalized_steps(transformer_model,
                                                                          original_shot_features[:, c, :],
                                                                          config['transformer']['time'],
                                                                          config['transformer']['numstep'])
                    predicted_prototype_eval_list.append(predicted_prototype_eval)

                    t, vlb_weights = timestep_sampler.sample(x_shot.shape[0], x_shot.device)
                    diffusion_loss, predicted_prototype_norm = diffusion.training_losses(transformer_model,
                                                                                    predicted_prototype_eval,
                                                                                    original_shot_features[:, c, :],
                                                                                    t)
                    predicted_prototype= predicted_prototype_norm[:, 0, :] + torch.mean(original_shot_features[:, c, :, :], dim=1)
                    predicted_prototype_list.append(torch.unsqueeze(predicted_prototype, dim=1))
                    diffusion_loss_list.append(diffusion_loss)
                predicted_prototypes = torch.cat(predicted_prototype_list, dim=1)
                with torch.no_grad():
                    diffusion_losses = torch.cat(diffusion_loss_list)
                    diffusion_losses_mean = torch.mean(diffusion_losses)
                    ori_logits, _, diffusion_logits, _, _ = model(x_shot, x_query, predicted_prototypes)
                    ori_logits = ori_logits.view(-1, n_train_way)
                    diffusion_logits = diffusion_logits.view(-1, n_train_way)
                    ce_loss = F.cross_entropy(diffusion_logits, label)
                    ori_acc = utils.compute_acc(ori_logits, label)
                    diffusion_acc = utils.compute_acc(diffusion_logits, label)

                    final_loss = ce_loss + config["loss_coff"] * diffusion_losses_mean
                aves[name_l].add(final_loss.item())
                aves[name_a].add(ori_acc)
                aves[name_dl].add(diffusion_losses_mean.item())
                aves[name_da].add(diffusion_acc)
                test_ori_acc.append(ori_acc)
                test_acc.append(diffusion_acc)
            test_acc_final = np.mean(test_acc)
            test_acc_ori_final = np.mean(test_ori_acc)

            if test_acc_final > best_test_acc:
                best_test_acc = test_acc_final
                best_test_epoch = epoch
            train_log = "Epoch: %d,  test_acc: %3f, test_acc_ori:%3f, best_test_acc: %3f, best_epoch: %d" % (
                epoch, test_acc_final, test_acc_ori_final, best_test_acc, best_test_epoch)
            log_file = open(args.meta_acc_logdir + "_" + str(config['n_shot']) + '/' + 'log.txt', "a")
            utils.print_and_log(log_file, train_log)


        # post
        if lr_scheduler_model is not None:
            lr_scheduler_model.step()
        if lr_scheduler_diffusion is not None:
            lr_scheduler_diffusion.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        writer.add_scalars('loss', {
            'train_final': aves['tl'],
            'train_diff': aves['tdl'],
            'train_ce': aves['tcel'],
            'val': aves['vl'],
            'vadl_diff': aves['vdl'],
        }, epoch)

        writer.add_scalars('acc', {
            'train_ori_acc': aves['toa'],
            'train_diff_acc': aves['tda'],
            'val_ori_acc': aves['va'],
            'val_diff_acc': aves['vda'],
        }, epoch)
        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer_model.state_dict(),
            'optimizer_diffusion': optimizer_diffusion.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,
            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),
            'diff_model_sd': transformer_model.state_dict(),
            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['vda'] > max_va:
            max_va = aves['vda']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_meta_mini_5.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--meta_acc_logdir', default="./meta_acc_logdir")
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
