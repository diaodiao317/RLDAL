import os
import gc
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torchvision.transforms as standard_transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import utils.transforms as extended_transforms
from data import cityscapes, camvid
from utils.logger import Logger
from utils.progressbar import progress_bar
from torch.optim.lr_scheduler import _LRScheduler
from utils.joint_transforms import generate_unsup_data
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

def compute_unsupervised_loss(predict, target, logits, unsup_weight= 0, strong_threshold=0.97):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()   # only count valid pixels

    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    
    loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
    if unsup_weight > 0:
        loss = loss * logits
        
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    return weighted_loss


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_and_load_optimizers(net, opt_choice, lr, wd,
                               momentum, ckpt_path, exp_name_toload, exp_name,
                               snapshot, checkpointer, load_opt,
                               policy_net=None, lr_dqn=0.0001, al_algorithm='random'):
    optimizerP = None
    opt_kwargs = {"lr": lr,
                  "weight_decay": wd,
                  "momentum": momentum
                  }
    opt_kwargs_rl = {"lr": lr_dqn,
                     "weight_decay": 0.001,
                     "momentum": momentum
                     }

    # optimizer = optim.SGD(
    #     params=filter(lambda p: p.requires_grad, net.parameters()),
    #     **opt_kwargs)
    optimizer = optim.SGD(params = net.parameters(), **opt_kwargs_rl)
    if al_algorithm == 'ralis' and policy_net is not None:
        if opt_choice == 'SGD':
            optimizerP = optim.SGD(
                params=filter(lambda p: p.requires_grad, policy_net.parameters()),
                **opt_kwargs_rl)
        elif opt_choice == 'RMSprop':
            optimizerP = optim.RMSprop(
                params=filter(lambda p: p.requires_grad, policy_net.parameters()),
                lr=lr_dqn)

    name = exp_name_toload if load_opt and len(exp_name_toload) > 0 else exp_name
    opt_path = os.path.join(ckpt_path, name, 'opt_' + snapshot)
    opt_policy_path = os.path.join(ckpt_path, name, 'opt_policy_' + snapshot)

    if (load_opt and len(exp_name_toload)) > 0 or (checkpointer and os.path.isfile(opt_path)):
        print('(Opt load) Loading net optimizer')
        optimizer.load_state_dict(torch.load(opt_path))

        if al_algorithm == 'ralis' and os.path.isfile(opt_policy_path):
            print('(Opt load) Loading policy optimizer')
            optimizerP.load_state_dict(torch.load(opt_policy_path))

    print ('Optimizers created')
    return optimizer, optimizerP


def get_logfile(ckpt_path, exp_name, checkpointer, snapshot, num_classes=19, log_name='log.txt'):
    log_columns = ['Epoch', 'Learning Rate', 'Train Loss', '(deprecated)',
                   'Valid Loss', 'Train Acc.', 'Valid Acc.',
                   'Train mean iu', 'Valid mean iu']
    for cl in range(num_classes):
        log_columns.append('iu_cl' + str(cl))
    best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'mean_iu': 0}
    curr_epoch = 0
    ##-- Check if log file exists --##
    if checkpointer:
        if os.path.isfile(os.path.join(ckpt_path, exp_name, log_name)):
            print('(Checkpointer) Log file ' + log_name + ' already exists, appending.')
            logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                            title=exp_name, resume=True)
            if 'best' in snapshot:
                curr_epoch = int(logger.resume_epoch)
            else:
                curr_epoch = logger.last_epoch
            best_record = {'epoch': int(logger.resume_epoch), 'val_loss': 1e10,
                           'mean_iu': float(logger.resume_jacc), 'acc': 0}
        else:
            print('(Checkpointer) Log file ' + log_name + ' did not exist before, creating')
            logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                            title=exp_name)
            logger.set_names(log_columns)

    else:
        print('(No checkpointer activated) Log file ' + log_name + ' created.')
        logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                        title=exp_name)
        logger.set_names(log_columns)
    return logger, best_record, curr_epoch


def get_training_stage(args):
    path = os.path.join(args.ckpt_path, args.exp_name,
                        'training_stage.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            stage = pickle.load(f)
    else:
        stage = None
    return stage


def set_training_stage(args, stage):
    path = os.path.join(args.ckpt_path, args.exp_name,
                        'training_stage.pkl')
    with open(path, 'wb') as f:
        pickle.dump(stage, f)


def evaluate(cm):
    # Compute metrics
    TP_perclass = cm.diagonal().astype('float32')
    FP_perclass = cm.sum(0) - TP_perclass  # False positives
    FN_perclass = cm.sum(1) - TP_perclass  # False negatives
    TN_perclass = cm.sum() - (TP_perclass + FP_perclass + FN_perclass)  # True negatives

    # Jaccard index for each class
    jaccard_perclass = TP_perclass / (cm.sum(1) + cm.sum(0) - TP_perclass)
    
    # Mean Jaccard index
    jaccard = np.mean(jaccard_perclass)
    
    # Accuracy
    accuracy = TP_perclass.sum() / cm.sum()
    
    # FDR and TPR
    FDR_perclass = FP_perclass / (FP_perclass + TP_perclass)
    TPR_perclass = TP_perclass / (TP_perclass + FN_perclass)
    
    # Mean FDR and TPR
    FDR = np.mean(FDR_perclass)
    TPR = np.mean(TPR_perclass)

    # Dice Coefficient for each class
    dice_perclass = 2 * TP_perclass / (2 * TP_perclass + FP_perclass + FN_perclass)
    
    # Mean Dice Coefficient
    dice = np.mean(dice_perclass)

    return accuracy, jaccard, jaccard_perclass, FDR, TPR, FDR_perclass, TPR_perclass, dice, dice_perclass


def confusion_matrix_pytorch(cm, output_flatten, target_flatten, num_classes):
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = cm[i, j] + ((output_flatten == i) * (target_flatten == j)).sum().type(torch.IntTensor).cuda()
    return cm


def compute_set_jacc(val_loader, net):
    """Compute accuracy, mean IoU and IoU per class on the provided set.
    :param dataset_target: Dataset
    :param net: Classification network
    :return: accuracy (float), iou (float), iou per class (list of floats)
    """
    net.eval()

    cm_py = torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes)).type(
        torch.IntTensor).cuda()
    for vi, data in enumerate(val_loader):
        inputs, gts_, _ = data
        with torch.no_grad():
            inputs = Variable(inputs).cuda()

        outputs, _ = net(inputs)

        if outputs.shape[2:] != gts_.shape[1:]:
            outputs = outputs[:, :, 0:min(outputs.shape[2], gts_.shape[1]), 0:min(outputs.shape[3], gts_.shape[2])]
            gts_ = gts_[:, 0:min(outputs.shape[2], gts_.shape[1]), 0:min(outputs.shape[3], gts_.shape[2])]
        predictions_py = outputs.data.max(1)[1].squeeze_(1)

        cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                         gts_.cuda().view(-1),
                                         val_loader.dataset.num_classes)

        del (outputs)
        del (predictions_py)
    acc, mean_iu, iu , FDR, TPR, FDR_perclass, TPR_perclass, dice, dice_perclass= evaluate(cm_py.cpu().numpy())
    return acc, mean_iu, iu

def train_new(train_loader,pretrain_loader, net, criterion, optimizer, supervised=False):
    net.train()
    train_loss = 0
    # cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
    #     torch.IntTensor).cuda()

    train_epoch = len(pretrain_loader)


    train_supervised_dataset = iter(pretrain_loader)
    train_semi_sup_dataset = iter(train_loader)

    for iter_idx in range(train_epoch):
        # ------------supervised data train----------------
        optimizer.zero_grad()
        data_sup = train_supervised_dataset.__next__()
        im_s, t_s_, _= data_sup
        t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
        # Get output of network
        outputs, _ = net(im_s)
        # Get segmentation maps
        # predictions_py = outputs.data.max(1)[1].squeeze_(1)
        sup_loss = criterion(outputs, t_s)

        #semi-supervised

        data_semi = train_semi_sup_dataset.__next__()
        
        im_semi, t_semi, _, _, _ = data_semi

        t_semi, im_semi = Variable(t_semi).cuda(), Variable(im_semi).cuda()
        outputs_semi, _ = net(im_semi)
        # Get segmentation maps
        # predictions_py = outputs_semi.data.max(1)[1].squeeze_(1)
        semi_loss = criterion(outputs_semi, t_semi)

        loss = sup_loss +  semi_loss
        # print(sup_loss,unsup_loss,semi_loss
        # loss = sup_loss + semi_loss
        # loss = sup_loss + unsup_loss
        train_loss += loss.item()
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
        optimizer.step()
        # cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
        #                                 t_semi.cuda().view(-1),
        #                                 train_loader.dataset.num_classes)

        progress_bar(iter_idx, len(train_loader), '[train loss %.5f]' % (
                train_loss / (iter_idx + 1)))

            # del (outputs)
            # del (loss)
            # gc.collect()
    # acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
    # print(' [train acc %.5f], [train iu %.5f]' % (
    #     acc, mean_iu))
    # return train_loss / (len(train_loader)), 0, acc, mean_iu
    return train_loss / (len(train_loader)), 0, 0, 0

def train(train_loader, net, criterion,cri2, optimizer, supervised=False):
    net.train()
    train_loss = 0
    loss_bs_list = []
    # cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
    #     torch.IntTensor).cuda()
    progress = tqdm(total=len(train_loader), desc='Train', leave=False)
    log_interval = 100
    print('train length:', len(train_loader))
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        if supervised:
            im_s, t_s_, _ = data
        else:
            im_s, t_s_, _, _, _ = data

        t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
        mask = t_s == train_loader.dataset.ignore_label
        # Get output of network
        outputs, _ = net(im_s)
        # Get segmentation maps
        predictions_py = outputs.data.max(1)[1].squeeze_(1)
        # testloss = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.ignore_label).cuda()
        # testloss = nn.CrossEntropyLoss(ignore_index = train_loader.dataset.ignore_label ).cuda()
        # loss_total = cri2(outputs,t_s)
        # print(loss_total)
        loss = criterion(outputs, t_s)
        loss_batch = cri2(outputs, t_s)
        # print(loss.shape)

        
        # test = testloss(outputs,t_s)
        # print(test.shape)
        # print(loss.sum() /(loss.numel() - mask.sum()),"===",test)
        
        loss_with_nan = torch.where(mask, torch.tensor(float('nan')).cuda(), loss)

        loss_with_nan = loss_with_nan.view(loss_with_nan.size(0),-1)
        loss_nanmean = torch.nanmean(loss_with_nan,dim=1)
        # print(loss_nanmean.shape,loss_nanmean)
        loss_cpu = loss_nanmean.cpu().detach().numpy()
        
        loss_bs_list.extend(loss_cpu)
        # print(loss_bs_list)
        # print(loss_nanmean)
        # print(loss_reshape.shape)
        # loss_bs = torch.mean(loss_nanmean,dim = 0)
        # loss_all = torch.mean(loss_bs)
        # print(loss_bs.shape,loss_bs)
        # loss_bs_list.append(loss_nanmean)
        
        # print('test',test)
        
        
        # print(outputs.shape)
        # print(loss_bs.shape)
        # print(loss_bs_list)
        # print(loss_all,loss_total)
        # print(loss_all.shape)
        # print(loss_bs.shape)
        # print(loss.shape)
        train_loss += loss_batch.item()
        
        loss_batch.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
        optimizer.step()

        # cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
        #                                  t_s_.cuda().view(-1),
        #                                  train_loader.dataset.num_classes)

        if ((i + 1) % log_interval == 0) or (i + 1 == len(train_loader)):
            avg_loss = train_loss / (i + 1)
            progress.set_postfix(loss=f"{avg_loss:.5f}")

        progress.update(1)

        # del (outputs)
        # del (loss)
        # gc.collect()
    progress.close()
    print(' ')
    # acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
    # print(' [train acc %.5f], [train iu %.5f]' % (
    #     acc, mean_iu))
    # return train_loss / (len(train_loader)), 0, acc, mean_iu
    return train_loss / (len(train_loader)), 0, 0, 0,loss_bs_list

def train_sup(train_loader, net, criterion, optimizer, supervised=False):
    net.train()
    train_loss = 0
    loss_bs_list = []
    # cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
    #     torch.IntTensor).cuda()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        if supervised:
            im_s, t_s_, _ = data
        else:
            im_s, t_s_, _, _, _ = data

        t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
        # print(t_s.shape)
        mask = t_s == train_loader.dataset.ignore_label
        # Get output of network
        outputs, _ = net(im_s)
        # Get segmentation maps
        predictions_py = outputs.data.max(1)[1].squeeze_(1)
        # testloss = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.ignore_label).cuda()
        # testloss = nn.CrossEntropyLoss(ignore_index = train_loader.dataset.ignore_label ).cuda()
        # loss_total = cri2(outputs,t_s)
        # print(loss_total)
        # print(t_s.shape)
        # print(outputs.shape)
        loss = criterion(outputs, t_s)

        train_loss += loss.item()
        
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
        optimizer.step()

        # cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
        #                                  t_s_.cuda().view(-1),
        #                                  train_loader.dataset.num_classes)

        progress_bar(i, len(train_loader), '[train loss %.5f]' % (
                train_loss / (i + 1)))

        # del (outputs)
        # del (loss)
        # gc.collect()
    print(' ')
    # acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
    # print(' [train acc %.5f], [train iu %.5f]' % (
    #     acc, mean_iu))
    # return train_loss / (len(train_loader)), 0, acc, mean_iu
    return train_loss / (len(train_loader)), 0, 0, 0
def train_final_ema(full_sup_loader,unlab_loader,train_loader, net,ema, criterion, optimizer, now_epo,total_epo,supervised=True):
    if now_epo < total_epo//5:
        print('now_unlab_training')
        net.train()
        ema.model.train()
        train_loss = 0
        cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()
        train_supervised_dataset = iter(full_sup_loader)
        train_semi_sup_dataset = iter(train_loader)
        train_unlab_dataset = iter(unlab_loader)
        train_epoch = len(unlab_loader)

        for iter_idx in range(train_epoch):
            # ------------supervised data train----------------
            optimizer.zero_grad()
            data_sup = train_supervised_dataset.__next__()
            if supervised:
                im_s, t_s_, _= data_sup
            else:
                im_s, t_s_, _, _, _ = data_sup
            t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
            # Get output of network
            outputs, _ = net(im_s)
            # Get segmentation maps
            predictions_py = outputs.data.max(1)[1].squeeze_(1)
            sup_loss = criterion(outputs, t_s)

            #unlab
            # data_ul = train_unlab_dataset.__next__()
            # if supervised:
                # im_ul, _, _ = data_ul
            # else:
                # im_ul, _, _, _, _ = data_ul
            # im_ul = Variable(im_ul).cuda()
            # with torch.no_grad():
            #     predmaps_ul,_ = ema.model(im_ul)
            #     predmaps_ul_xl = F.interpolate(predmaps_ul, size=im_ul.shape[2:], mode='bilinear',
            #                                         align_corners=True)
            #     pseudo_logits, pseudo_labels = torch.max(torch.softmax(predmaps_ul_xl, dim=1), dim=1)

            # predmaps_ul,_ = net(im_ul)
            # predmaps_ul_xl = F.interpolate(predmaps_ul, size=im_ul.shape[2:], mode='bilinear',
            #                                     align_corners=True)
            
            # trans_data, trans_label, trans_logits = generate_unsup_data(im_ul, pseudo_labels, pseudo_logits)
            
            # predmaps_trans_ul,_ = net(trans_data)
            # predmaps_trans_ul = F.interpolate(predmaps_trans_ul, size=im_ul.shape[2:], mode='bilinear',
                                                # align_corners=True)
            
            # unsup_loss = compute_unsupervised_loss(predict=predmaps_trans_ul, 
            #                                             target= trans_label, 
            #                                             logits=trans_logits,
            #                                             unsup_weight=0,
            #                                             strong_threshold=0.97)

            #semi-supervised
            data_semi = train_semi_sup_dataset.__next__()
            if supervised:
                im_semi, t_semi, _, _, _ = data_semi
            else:
                im_semi, t_semi, _, _, _ = data_semi
            t_semi, im_semi = Variable(t_semi).cuda(), Variable(im_semi).cuda()
            outputs_semi, _ = net(im_semi)
            # Get segmentation maps
            predictions_py = outputs_semi.data.max(1)[1].squeeze_(1)
            semi_loss = criterion(outputs_semi, t_semi)

            loss = sup_loss +  semi_loss
            # print(sup_loss,unsup_loss,semi_loss)
            # loss = sup_loss + semi_loss
            # loss = sup_loss + unsup_loss
            train_loss += loss.item()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
            optimizer.step()
            ema.update(net)
            # cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
            #                                 t_semi.cuda().view(-1),
            #                                 train_loader.dataset.num_classes)
            if iter_idx% 50 == 0:
                print('train loss:', iter_idx, train_loss / (iter_idx + 1))

            # progress_bar(iter_idx, len(train_loader), '[train loss %.5f]' % (
            #         train_loss / (iter_idx + 1)))
        # net.train()
        # train_loss = 0
        # cm_py = torch.zeros((full_sup_loader.dataset.num_classes, full_sup_loader.dataset.num_classes)).type(
        #     torch.IntTensor).cuda()
        # for i, data in enumerate(full_sup_loader):
            
#             optimizer.zero_grad()
#             if supervised:
#                 im_s, t_s_, _ = data
#             else:
#                 im_s, t_s_, _, _, _ = data

#             t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
#             # Get output of network
#             outputs, _ = net(im_s)
#             # Get segmentation maps
#             predictions_py = outputs.data.max(1)[1].squeeze_(1)
#             loss = criterion(outputs, t_s)
#             train_loss += loss.item()

#             loss.backward()
#             optimizer.step()
#             # cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
#             #                                 t_s_.cuda().view(-1),
#             #                                 full_sup_loader.dataset.num_classes)

#             progress_bar(i, len(train_loader), '[train loss %.5f]' % (
#                     train_loss / (i + 1)))

            # del (outputs)
            # del (loss)
            # gc.collect()
    else:
        print('now_unlab_training')
        net.train()
        ema.model.train()
        train_loss = 0
        cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()
        train_supervised_dataset = iter(full_sup_loader)
        train_semi_sup_dataset = iter(train_loader)
        train_unlab_dataset = iter(unlab_loader)
        train_epoch = len(unlab_loader)

        for iter_idx in range(train_epoch):
            # ------------supervised data train----------------
            optimizer.zero_grad()
            data_sup = train_supervised_dataset.__next__()
            if supervised:
                im_s, t_s_, _= data_sup
            else:
                im_s, t_s_, _, _, _ = data_sup
            t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
            # Get output of network
            outputs, _ = net(im_s)
            # Get segmentation maps
            predictions_py = outputs.data.max(1)[1].squeeze_(1)
            sup_loss = criterion(outputs, t_s)

            #unlab
            data_ul = train_unlab_dataset.__next__()
            if supervised:
                im_ul, _, _ = data_ul
            else:
                im_ul, _, _, _, _ = data_ul
            im_ul = Variable(im_ul).cuda()
            with torch.no_grad():
                predmaps_ul,_ = ema.model(im_ul)
                predmaps_ul_xl = F.interpolate(predmaps_ul, size=im_ul.shape[2:], mode='bilinear',
                                                    align_corners=True)
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(predmaps_ul_xl, dim=1), dim=1)

            # predmaps_ul,_ = net(im_ul)
            # predmaps_ul_xl = F.interpolate(predmaps_ul, size=im_ul.shape[2:], mode='bilinear',
            #                                     align_corners=True)
            
            trans_data, trans_label, trans_logits = generate_unsup_data(im_ul, pseudo_labels, pseudo_logits)
            
            predmaps_trans_ul,_ = net(trans_data)
            predmaps_trans_ul = F.interpolate(predmaps_trans_ul, size=im_ul.shape[2:], mode='bilinear',
                                                align_corners=True)
            
            unsup_loss = compute_unsupervised_loss(predict=predmaps_trans_ul, 
                                                        target= trans_label, 
                                                        logits=trans_logits,
                                                        unsup_weight=0,
                                                        strong_threshold=0.97)

            #semi-supervised
            data_semi = train_semi_sup_dataset.__next__()
            if supervised:
                im_semi, t_semi, _, _, _ = data_semi
            else:
                im_semi, t_semi, _, _, _ = data_semi
            t_semi, im_semi = Variable(t_semi).cuda(), Variable(im_semi).cuda()
            outputs_semi, _ = net(im_semi)
            # Get segmentation maps
            predictions_py = outputs_semi.data.max(1)[1].squeeze_(1)
            semi_loss = criterion(outputs_semi, t_semi)

            loss = sup_loss + unsup_loss + semi_loss
            # print(sup_loss,unsup_loss,semi_loss)
            # loss = sup_loss + semi_loss
            # loss = sup_loss + unsup_loss
            train_loss += loss.item()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
            optimizer.step()
            ema.update(net)
            # cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
            #                                 t_semi.cuda().view(-1),
            #                                 train_loader.dataset.num_classes)

            if iter_idx% 50 == 0:
                print('train loss:', iter_idx, train_loss / (iter_idx + 1))

            # del (outputs)
            # del (loss)
            # gc.collect()

    print(' ')
    # acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
    # print(' [train acc %.5f], [train iu %.5f]' % (
    #     acc, mean_iu))
    return train_loss / (len(train_loader)), 0, 0, 0
    
    

def pretrain(train_loader,unlab_loader, ema, net, criterion, optimizer,now_epoch,total_pretrain_epo, supervised=False):
    if now_epoch < total_pretrain_epo//5:
        net.train()
        train_loss = 0
        cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if supervised:
                im_s, t_s_, _ = data
            else:
                im_s, t_s_, _, _, _ = data

            t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
            # Get output of network
            outputs, _ = net(im_s)
            # Get segmentation maps
            predictions_py = outputs.data.max(1)[1].squeeze_(1)
            loss = criterion(outputs, t_s)
            train_loss += loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
            optimizer.step()
            cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                            t_s_.cuda().view(-1),
                                            train_loader.dataset.num_classes)

            progress_bar(i, len(train_loader), '[train loss %.5f]' % (
                    train_loss / (i + 1)))

            # del (outputs)
            # del (loss)
            # gc.collect()
    else:
        print('now_unlab_training')
        net.train()
        ema.model.train()
        train_loss = 0
        cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()
        train_supervised_dataset = iter(train_loader)
        train_unlab_dataset = iter(unlab_loader)
        train_epoch = len(unlab_loader)

        for iter_idx in range(train_epoch):
            # ------------supervised data train----------------
            optimizer.zero_grad()
            data_sup = train_supervised_dataset.__next__()
            if supervised:
                im_s, t_s_, _ = data_sup
            else:
                im_s, t_s_, _, _, _ = data_sup
            t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
            # Get output of network
            outputs, _ = net(im_s)
            # Get segmentation maps
            predictions_py = outputs.data.max(1)[1].squeeze_(1)
            sup_loss = criterion(outputs, t_s)

            #unlab
            data_ul = train_unlab_dataset.__next__()
            if supervised:
                im_ul, _, _ = data_ul
            else:
                im_ul, _, _, _, _ = data_ul
            im_ul = Variable(im_ul).cuda()
            with torch.no_grad():
                predmaps_ul,_ = ema.model(im_ul)
                predmaps_ul_xl = F.interpolate(predmaps_ul, size=im_ul.shape[2:], mode='bilinear',
                                                   align_corners=True)
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(predmaps_ul_xl, dim=1), dim=1)
                
            predmaps_ul,_ = net(im_ul)
            predmaps_ul_xl = F.interpolate(predmaps_ul, size=im_ul.shape[2:], mode='bilinear',
                                               align_corners=True)
            unsup_loss = compute_unsupervised_loss(predict=predmaps_ul_xl, 
                                                       target= pseudo_labels, 
                                                       logits=pseudo_logits,
                                                       unsup_weight=0,
                                                       strong_threshold=0.97)

            loss = sup_loss + unsup_loss
            train_loss += loss.item()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
            optimizer.step()
            ema.update(net)
            cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                            t_s_.cuda().view(-1),
                                            train_loader.dataset.num_classes)

            progress_bar(iter_idx, len(train_loader), '[train loss %.5f]' % (
                    train_loss / (iter_idx + 1)))

            # del (outputs)
            # del (loss)
            # gc.collect()

    print(' ')
    acc, mean_iu, iu, FDR, TPR, FDR_perclass, TPR_perclass, dice, dice_perclass = evaluate(cm_py.cpu().numpy())
    print(' [train acc %.5f], [train iu %.5f],[FDR %.5f],[TPR %.5f],[dice %.5f] ' % (
        acc, mean_iu, FDR,TPR,dice))
    print(dice_perclass)
    return train_loss / (len(train_loader)), 0, acc, mean_iu

def train_ema(train_loader, net, ema, criterion, optimizer, supervised=False):
    net.train()
    ema.model.eval()
    
    train_loss = 0
    cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
        torch.IntTensor).cuda()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        if supervised:
            im_s, t_s_, _ = data
        else:
            im_s, t_s_, _, _, _ = data

        t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
        # Get output of network
        outputs, _ = net(im_s)
        # Get segmentation maps
        predictions_py = outputs.data.max(1)[1].squeeze_(1)
        loss = criterion(outputs, t_s)
        train_loss += loss.item()

        loss.backward()
        ema.update(net)
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
        optimizer.step()

        cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                         t_s_.cuda().view(-1),
                                         train_loader.dataset.num_classes)

        progress_bar(i, len(train_loader), '[train loss %.5f]' % (
                train_loss / (i + 1)))

        del (outputs)
        del (loss)
        gc.collect()
    print(' ')
    acc, mean_iu, iu , FDR, TPR, FDR_perclass, TPR_perclass, dice, dice_perclass= evaluate(cm_py.cpu().numpy())
    print(' [train acc %.5f], [train iu %.5f]' % (
        acc, mean_iu))
    print(' [train acc %.5f], [train iu %.5f],[FDR %.5f],[TPR %.5f],[dice %.5f] ' % (acc, mean_iu, FDR,TPR,dice))
    print('dice_perclass:',dice_perclass)
    return train_loss / (len(train_loader)), 0, acc, mean_iu



def validate(val_loader, net, criterion, optimizer, epoch, best_record, args, final_final_test=False):
    net.eval()

    val_loss = 0
    cm_py = torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes)).type(
        torch.IntTensor).cuda()
    for vi, data in enumerate(val_loader):
        inputs, gts_, _ = data
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            gts = Variable(gts_).cuda()
        # print('input',inputs.shape)
        outputs, _ = net(inputs)
        # print('outputs',outputs.shape)
        # Make sure both output and target have the same dimensions
        if outputs.shape[2:] != gts.shape[1:]:
            outputs = outputs[:, :, 0:min(outputs.shape[2], gts.shape[1]), 0:min(outputs.shape[3], gts.shape[2])]
            gts = gts[:, 0:min(outputs.shape[2], gts.shape[1]), 0:min(outputs.shape[3], gts.shape[2])]
        predictions_py = outputs.data.max(1)[1].squeeze_(1)
        # print(gts.shape)
        # print(outputs.shape)
        loss = criterion(outputs, gts)
        # loss_reshape = loss.view(loss.size(0),-1)
        # print(loss_reshape.shape)
        # loss_bs = torch.mean(loss_reshape,dim = 1)
        # loss = torch.mean(loss_bs)
        
        vl_loss = loss.item()
        val_loss += (vl_loss)

        cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                         gts_.cuda().view(-1),
                                         val_loader.dataset.num_classes)

        len_val = len(val_loader)
        # progress_bar(vi, len_val, '[val loss %.5f]' % (
        #         val_loss / (vi + 1)))

        del (outputs)
        del (vl_loss)
        del (loss)
        del (predictions_py)
    acc, mean_iu, iu , FDR, TPR, FDR_perclass, TPR_perclass, dice, dice_perclass= evaluate(cm_py.cpu().numpy())
    print(' ')
    print('iu_class:',iu)
    print(' [val acc %.5f], [val iu %.5f],[FDR %.5f],[TPR %.5f],[dice %.5f]' % (
        acc, mean_iu,FDR,TPR,dice))
    print('iu_class:',iu)
    print('dice_class:',dice_perclass)

    if not final_final_test:
        if mean_iu > best_record['mean_iu']:
            best_record['val_loss'] = val_loss / len(val_loader)
            best_record['epoch'] = epoch
            best_record['acc'] = acc
            best_record['iu'] = iu
            best_record['mean_iu'] = mean_iu

            torch.save(net.cpu().state_dict(),
                       os.path.join(args.ckpt_path, args.exp_name,
                                    'best_jaccard_val.pth'))
            net.cuda()
            torch.save(optimizer.state_dict(),
                       os.path.join(args.ckpt_path, args.exp_name,
                                    'opt_best_jaccard_val.pth'))

        ## Save checkpoint every epoch
        torch.save(net.cpu().state_dict(),
                   os.path.join(args.ckpt_path, args.exp_name,
                                'last_jaccard_val.pth'))
        net.cuda()
        torch.save(optimizer.state_dict(),
                   os.path.join(args.ckpt_path, args.exp_name,
                                'opt_last_jaccard_val.pth'))

        print(
                'best record: [val loss %.5f], [acc %.5f], [mean_iu %.5f],'
                ' [epoch %d]' % (best_record['val_loss'], best_record['acc'],
                                 best_record['mean_iu'], best_record['epoch']))

    print('----------------------------------------')

    return val_loss / len(val_loader), acc, mean_iu, iu, best_record


def test(val_loader, net, criterion):
    net.eval()

    val_loss = 0
    cm_py = torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes)).type(
        torch.IntTensor).cuda()
    for vi, data in enumerate(val_loader):
        inputs, gts_, _ = data
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            gts = Variable(gts_).cuda()

        outputs, _ = net(inputs)
        predictions_py = outputs.data.max(1)[1].squeeze_(1)
        loss = criterion(outputs, gts)
        vl_loss = loss.item()
        val_loss += (vl_loss)

        cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                         gts_.cuda().view(-1),
                                         val_loader.dataset.num_classes)

        len_val = len(val_loader)
        progress_bar(vi, len_val, '[val loss %.5f]' % (
                val_loss / (vi + 1)))

        del (outputs)
        del (vl_loss)
    acc, mean_iu, iu, FDR, TPR, FDR_perclass, TPR_perclass, dice, dice_perclass = evaluate(cm_py.cpu().numpy())
    print(' ')
    print(' [val acc %.5f], [val iu %.5f]' % (
        acc, mean_iu))

    return val_loss / len(val_loader), acc, mean_iu, iu


def final_test(args, net, criterion):
    # Load best checkpoint for segmentation network
    net_checkpoint_path = os.path.join(args.ckpt_path, args.exp_name, 'best_jaccard_val.pth')
    if os.path.isfile(net_checkpoint_path):
        print('(Final test) Load best checkpoint for segmentation network!')
        net_dict = torch.load(net_checkpoint_path)
        if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in net_dict.items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            net_dict = new_state_dict
        net.load_state_dict(net_dict)
    net.eval()

    # Prepare data transforms
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()

    if 'camvid' in args.dataset:
        val_set = camvid.Camvid('fine', 'test' if test else 'val',
                                data_path=args.data_path,
                                joint_transform=None,
                                transform=input_transform,
                                target_transform=target_transform)
        val_loader = DataLoader(val_set,
                                batch_size=4,
                                num_workers=2, shuffle=False)
    else:
        val_set = cityscapes.CityScapes('fine', 'val',
                                        data_path=args.data_path,
                                        joint_transform=None,
                                        transform=input_transform,
                                        target_transform=target_transform)
        val_loader = DataLoader(val_set,
                                batch_size=args.val_batch_size,
                                num_workers=2, shuffle=False)
    print('Starting test...')
    vl_loss, val_acc, val_iu, iu_xclass = test(val_loader, net, criterion)
    ## Append info to logger
    info = [vl_loss, val_acc, val_iu]
    for cl in range(val_loader.dataset.num_classes):
        info.append(iu_xclass[cl])
    rew_log = open(os.path.join(args.ckpt_path, args.exp_name, 'test_results.txt'), 'a')
    for inf in info:
        rew_log.write("%f," % (inf))
    rew_log.write("\n")
