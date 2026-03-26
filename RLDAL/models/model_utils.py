import time
import math
import numpy as np
import os
import random
from scipy.stats import entropy
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from models.query_network import QueryNetworkDQN
from models.fpn import FPN50
from models.deeplabv3_mb import DeepLabv3Plus_MB
from models.deeplabv3_resnet import DeepLabv3Plus_RN
from models.mobilenetv2 import mobilenet_v2
from models.resnet import resnet50_d8
from utils.final_utils import get_logfile
from utils.progressbar import progress_bar
from models.unet_model import UNet
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 200
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
 
    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def reset(self, model):
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = param.data
        self.step = 0

    def update(self,model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1


def create_models(dataset, al_algorithm, region_size):
    """Returns segmentation model FPN with backbone ResNet50
    :param dataset: (str) Which dataset to use.
    :param al_algorithm: (str) Active learning algorithm, expected 'ralis'.
    :param region_size: (tuple) Size of regions, [width, height]
    :return: segmentation network, query network, target network (same construction as query network)
    """

    # Segmentation network
    n_cl = 19
    if 'TUI' in dataset:
        n_cl = 3
    if 'KVASIR' in dataset:
        n_cl = 2
    if 'TN3K' in dataset:
        n_cl = 2
    if 'ACDC' in dataset:
        n_cl = 4
    # net = UNet(n_channels=3,n_classes=n_cl).cuda()

    mbv2 = mobilenet_v2(pretrained=True).cuda()
    net = DeepLabv3Plus_MB(network_mbv2=mbv2,num_classes=n_cl).cuda()
    # res50 = resnet50_d8(pretrained=True).cuda()
    # net = DeepLabv3Plus_RN(network_rn=res50,num_classes=n_cl).cuda()


    print('Model has ' + str(count_parameters(net)))

    # Query network (and target network for DQN)
    input_size = [(n_cl + 1) + 3 * 64, (n_cl + 1) + 3 * 64]
    if al_algorithm == 'ralis':
        image_size = [2048, 1024]
        if 'TUI' in dataset:
            image_size = [256,256]
        if 'KVASIR' in dataset:
            image_size = [256,256]
        if 'TN3K' in dataset:
            image_size = [256,256]
        if 'ACDC' in dataset:
            image_size = [256,256]
        indexes_full_state = 10 * (image_size[0] // region_size[0]) * (image_size[1] // region_size[1])

        policy_net = QueryNetworkDQN(input_size=input_size[0], input_size_subset=input_size[1],
                                     indexes_full_state=indexes_full_state).cuda()
        target_net = QueryNetworkDQN(input_size=input_size[0], input_size_subset=input_size[1],
                                     indexes_full_state=indexes_full_state).cuda()
        print('Policy network has ' + str(count_parameters(policy_net)))
    else:
        policy_net = None
        target_net = None

    print('Models created!')
    return net, policy_net, target_net




def count_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def load_models(net, load_weights, exp_name_toload, snapshot,
                exp_name, ckpt_path, checkpointer, exp_name_toload_rl='',
                policy_net=None, target_net=None, test=False, dataset='ACDC', al_algorithm='ralis'):
    """Load model weights.
    :param net: Segmentation network
    :param load_weights: (bool) True if segmentation network is loaded from pretrained weights in 'exp_name_toload'
    :param exp_name_toload: (str) Folder where to find pre-trained segmentation network's weights.
    :param snapshot: (str) Name of the checkpoint.
    :param exp_name: (str) Experiment name.
    :param ckpt_path: (str) Checkpoint name.
    :param checkpointer: (bool) If True, load weights from the same folder.
    :param exp_name_toload_rl: (str) Folder where to find trained weights for the query network (DQN). Used to test
    query network.
    :param policy_net: Policy network.
    :param target_net: Target network.
    :param test: (bool) If True and al_algorithm='ralis' and there exists a checkpoint in 'exp_name_toload_rl',
    we will load checkpoint for trained query network (DQN).
    :param dataset: (str) Which dataset.
    :param al_algorithm: (str) Which active learning algorithm.
    """

    policy_path = os.path.join(ckpt_path, exp_name_toload_rl, 'policy_' + snapshot)
    net_path = os.path.join(ckpt_path, exp_name_toload, 'best_jaccard_val.pth')

    policy_checkpoint_path = os.path.join(ckpt_path, exp_name, 'policy_' + snapshot)
    net_checkpoint_path = os.path.join(ckpt_path, exp_name, 'last_jaccard_val.pth')

    ####------ Load policy (RL) from one folder and network from another folder ------####
    if al_algorithm == 'ralis' and test and os.path.isfile(policy_path):
        print ('(RL and TEST) Testing policy from another experiment folder!')
        policy_net.load_state_dict(torch.load(policy_path))
        # Load pre-trained segmentation network from another folder (best checkpoint)
        if load_weights and len(exp_name_toload) > 0:
            print('Loading pre-trained segmentation network (best checkpoint).')
            net_dict = torch.load(net_path)
            if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in net_dict.items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                net_dict = new_state_dict
            net.load_state_dict(net_dict)
            net.cuda()

    #     if checkpointer:  # In case the experiment is interrupted, load most recent segmentation network
    #         if os.path.isfile(net_checkpoint_path):
    #             print('(RL and TEST) Loading checkpoint for segmentation network!')
    #             net_dict = torch.load(net_checkpoint_path)
    #             if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
    #                 from collections import OrderedDict
    #                 new_state_dict = OrderedDict()
    #                 for k, v in net_dict.items():
    #                     name = k[7:]  # remove module.
    #                     new_state_dict[name] = v
    #                 net_dict = new_state_dict
    #             net.load_state_dict(net_dict)
    # else:
    #     ####------ Load experiment from another folder ------####
    #     if load_weights and len(exp_name_toload) > 0:
    #         print('(From another exp) training resumes from best_jaccard_val.pth')
    #         net_dict = torch.load(net_path)
    #         if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
    #             from collections import OrderedDict
    #             new_state_dict = OrderedDict()
    #             for k, v in net_dict.items():
    #                 name = k[7:]  # remove module.
    #                 new_state_dict[name] = v
    #             net_dict = new_state_dict
    #         net.load_state_dict(net_dict)

    #     ####------ Resume experiment ------####
    #     if checkpointer:
    #         ##-- Check if weights exist --##
    #         if os.path.isfile(net_checkpoint_path):
    #             print('(Checkpointer) training resumes from last_jaccard_val.pth')
    #             net_dict = torch.load(net_checkpoint_path)
    #             if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
    #                 from collections import OrderedDict
    #                 new_state_dict = OrderedDict()
    #                 for k, v in net_dict.items():
    #                     name = k[7:]  # remove module.
    #                     new_state_dict[name] = v
    #                 net_dict = new_state_dict
    #             net.load_state_dict(net_dict)
    #             if al_algorithm == 'ralis' and os.path.isfile(policy_checkpoint_path) \
    #                     and policy_net is not None and target_net is not None:
    #                 print('(Checkpointer RL) training resumes from ' + snapshot)
    #                 policy_net.load_state_dict(torch.load(policy_checkpoint_path))
    #                 policy_net.cuda()
    #                 target_net.load_state_dict(torch.load(policy_checkpoint_path))
    #                 target_net.cuda()
    #         else:
    #             print('(Checkpointer) Training starts from scratch')

    ####------ Get log file ------####
    if al_algorithm == 'ralis':
        logger = None
        best_record = None
        curr_epoch = None
    else:
        num_classes = 19
        if 'TUI' in dataset:
            num_classes = 3
        if 'KVASIR' in dataset:
            num_classes = 2
        if 'TN3K' in dataset:
            num_classes = 2
        if 'ACDC' in dataset:
            num_classes = 4

        logger, best_record, curr_epoch = get_logfile(ckpt_path, exp_name, checkpointer, snapshot,
                                                      num_classes=num_classes)

    return logger, curr_epoch, best_record


def get_region_candidates(candidates, train_set, num_regions=2):
    """Get region candidates function.
    :param candidates: (list) randomly sampled image indexes for images that contain unlabeled regions.
    :param train_set: Training set.
    :param num_regions: Number of regions to take as possible regions to be labeled.
    :return: candidate_regions: List of tuples (int(Image index), int(width_coord), int(height_coord)).
        The coordinate is the left upper corner of the region.
    """
    s = time.time()
    print('Getting region candidates...')
    total_regions = num_regions
    candidate_regions = []
    #### --- Get candidate regions --- ####
    counter_regions = 0
    available_regions = train_set.get_num_unlabeled_regions()
    rx, ry = train_set.get_unlabeled_regions()
    while counter_regions < total_regions and (total_regions - counter_regions) <= available_regions:
        index_ = np.random.choice(len(candidates))
        index = candidates[index_]
        num_regions_left = train_set.get_num_unlabeled_regions_image(int(index))
        if num_regions_left > 0:
            counter_x, counter_y = train_set.get_random_unlabeled_region_image(int(index))
            candidate_regions.append((int(index), counter_x, counter_y))
            available_regions -= 1
            counter_regions += 1
            if num_regions_left == 1:
                candidates.pop(int(index_))
        else:
            print ('This image has no more unlabeled regions!')

    train_set.set_unlabeled_regions(rx, ry)
    print ('Regions candidates indexed! Time elapsed: ' + str(time.time() - s))
    print ('Candidate regions are ' + str(counter_regions))
    return candidate_regions


def compute_state(args, net, region_candidates, candidate_set, train_set, num_groups=5, reg_sz=(128, 128)):
    """Computes the state of a given candidate pool of size N.
    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param net: Segmentation network.
    :param region_candidates: (list) of region candidates to be potentially labeled in this iteration
    :param candidate_set: (list) of image indexes to be potentially labeled in this iteration
    :param train_set: Training dataset
    :param num_groups: (int) Number of regions to label at this iteration
    :param reg_sz: (tuple) region size.
    :return: a Torch tensor with the state-action representation, region candidates rearranged to match
    the order in state-action representation
    """
    s = time.time()
    print ('Computing state...')
    net.eval()
    device = next(net.parameters()).device
    state = []
    state_ent = []
    old_candidate = None
    predictions_py = None
    predictions_py_prob = None
    pred_py = None
    ent = None

    region_candidates.sort()
    for candidates in region_candidates:
        if not candidates[0] == old_candidate:
            del (pred_py)
            del (predictions_py_prob)
            del (predictions_py)
            del (ent)
            old_candidate = candidates[0]
            inputs, gts, _, _ = candidate_set.get_specific_item(candidates[0])
            inputs, gts = Variable(inputs).to(device), Variable(gts).to(device)
            with torch.no_grad():
                outputs, _ = net(inputs.unsqueeze(0))

                # Softmax and prediction maps
                pred_soft = F.softmax(outputs, dim=1)
                log_soft = torch.log_softmax(outputs, dim=1)
                ent = -torch.sum(pred_soft * log_soft, dim=1).detach()

                pred_py = pred_soft.detach()
            del (outputs)
            del (pred_soft)
            pred_py = pred_py.max(1)
            predictions_py = pred_py[1].squeeze_(1).cpu().type(torch.FloatTensor)
            predictions_py_prob = pred_py[0].squeeze_(1).cpu().type(torch.FloatTensor)

        pred_region = predictions_py[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                      int(candidates[1]):int(candidates[1]) + reg_sz[0]]
        pred_region_prob = predictions_py_prob[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                           int(candidates[1]):int(candidates[1]) + reg_sz[0]]

        ent_region_ent = ent[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                     int(candidates[1]):int(candidates[1]) + reg_sz[0]].mean()
        sample_stats_ent = []
        sample_stats_ent.append(ent_region_ent.item())

        ent_region = ent[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                     int(candidates[1]):int(candidates[1]) + reg_sz[0]]

        # Convert 2D maps into vector representation
        sample_stats = create_feature_vector_3H_region_kl_sim(pred_region, ent_region, train_set,
                                                              num_classes=train_set.num_classes, reg_sz=reg_sz)

        state.append(torch.tensor(sample_stats, device=device, dtype=torch.float32).unsqueeze(0))
        state_ent.append(torch.tensor(sample_stats_ent, device=device, dtype=torch.float32).unsqueeze(0))
        del (pred_region)
        del (pred_region_prob)
        del (ent_region)
    del (pred_py)
    del (predictions_py)
    del (predictions_py_prob)
    state = torch.cat(state, dim=0).float()
    state_ent = torch.cat(state_ent, dim=0).float()

    # Shuffle vector so that we do not end up with regions from the same image in the same pool
    randperm = torch.randperm(state.size()[0], device=device)
    state = state[randperm]
    state_ent = state_ent[randperm]
    print(state_ent.shape)
    region_candidates = np.array(region_candidates)[randperm.cpu()]
    # region_candidates = np.array(region_candidates)
    state = state.view(num_groups, state.size()[0] // num_groups, state.size()[1])
    # print(num_groups,state_ent.size()[0] // num_groups, state_ent.size()[1])
    state_ent = state_ent.view(num_groups, state_ent.size()[0] // num_groups, state_ent.size()[1])
    region_candidates = np.reshape(region_candidates,
                                   (num_groups, region_candidates.shape[0] // num_groups, region_candidates.shape[1]))
    state = state  # [groups, cand_regions, channels, reg_size, reg_size]
    state_ent = state_ent
    # print(state.shape)
    # Adding KL terms for action representation
    state = add_kl_pool2(state, n_cl=train_set.num_classes)
    # Add fixed part of state (from state subset)
    state_subset = []
    for index in range(0, len(candidate_set.state_subset)):
        inputs, gts, _, _, regions = candidate_set.get_subset_state(index)
        inputs, gts = Variable(inputs).to(device), Variable(gts).to(device)
        with torch.no_grad():
            outputs, _ = net(inputs.unsqueeze(0))
            pred_soft = F.softmax(outputs, dim=1)
            log_soft = torch.log_softmax(outputs, dim=1)
            ent = -torch.sum(pred_soft * log_soft, dim=1).detach()
            pred_py = pred_soft.detach()
        del (outputs)
        del (pred_soft)
        del (log_soft)
        pred_py = pred_py.max(1)
        predictions_py = pred_py[1].squeeze_(1).cpu().type(torch.FloatTensor)
        predictions_py_prob = pred_py[0].squeeze_(1).cpu().type(torch.FloatTensor)

        for reg in regions:
            pred_region_prob = predictions_py_prob[0, int(reg[1]):int(reg[1]) + reg_sz[1],
                               int(reg[0]):int(reg[0]) + reg_sz[0]]
            pred_region = predictions_py[0, int(reg[1]):int(reg[1]) + reg_sz[1],
                          int(reg[0]):int(reg[0]) + reg_sz[0]]

            ent_region = ent[0, int(reg[1]):int(reg[1]) + reg_sz[1], int(reg[0]):int(reg[0]) + reg_sz[0]]
            # Convert 2D maps into vector representations
            sample_stats = create_feature_vector_3H_region_kl(pred_region, ent_region,
                                                              num_classes=train_set.num_classes,
                                                              reg_sz=reg_sz)
            state_subset.append(torch.tensor(sample_stats, device=device).unsqueeze(0))
            del (pred_region)
            del (pred_region_prob)
            del (ent_region)

        del (ent)
        del (pred_py)
        del (predictions_py)
        del (predictions_py_prob)
    state_subset = torch.cat(state_subset, dim=0).float()
    all_state = {'subset': state_subset, 'pool': state}
    print ('State computed! Time elapsed: ' + str(time.time() - s))
    return all_state, region_candidates, state_ent


def select_action(args, policy_net, all_state,state_ent, steps_done, test=False):
    """We select the action: index of the image to label.
    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param policy_net: policy network.
    :param all_state: (torch.Variable) Torch tensor containing the state-action representation.
    :param steps_done: (int) Number of images labeled so far.
    :param test: (bool) Whether we are testing the DQN or training it.
    :return: Action (indexes of the regions to label), updated step counter.
    """
    print(args.al_algorithm,'======================')
    if args.al_algorithm != 'ralis':
        raise ValueError("Only 'ralis' is supported in this codebase")

    state = all_state['pool']
    state_subset = all_state['subset']
    device = state.device if torch.is_tensor(state) else torch.device('cuda')
    ent = 0
    reset_net = False
    policy_net.eval()
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    q_val_ = []
    if sample > eps_threshold or test:
        print ('Action selected with DQN!')
        with torch.no_grad():
            # Splitting state to fit it in GPU memory
            state_subset_device = state_subset.to(device) if state_subset is not None else None
            for i in range(0, state.size()[0], 16):
                if state_subset_device is not None and len(state_subset_device.shape) == 2:
                    q_val_.append(policy_net(state[i:i + 16].to(device),
                                             state_subset_device.unsqueeze(0).repeat(state[i:i + 16].size()[0], 1,
                                                                                      1)))
                else:
                    q_val_.append(policy_net(state[i:i + 16].to(device),
                                             state_subset_device.unsqueeze(0).repeat(state[i:i + 16].size()[0], 1, 1,
                                                                                      1)))

            q_val_ = torch.cat(q_val_)
            state_ent = state_ent.to(device).float().squeeze()
            state_avg = state_ent.mean(dim=1, keepdim=True)
            flag = state_ent > state_avg
            state_ent = state_ent * flag

            # Filter: drop entropy==0 and keep only top 1/3 (ceiling) per group
            keep_mask = torch.zeros_like(state_ent, dtype=torch.bool)
            skip_rows = []
            for i in range(state_ent.size(0)):
                ent_row = state_ent[i]
                pos_idx = torch.nonzero(ent_row > 0, as_tuple=False).view(-1)
                pos_count = pos_idx.numel()
                if pos_count == 0:
                    skip_rows.append(i)
                    continue
                keep_k = max(1, math.ceil(pos_count / 3))
                _, topk_idx = torch.topk(ent_row[pos_idx], k=keep_k, largest=True, sorted=False)
                keep_indices = pos_idx[topk_idx]
                keep_mask[i, keep_indices] = True
            state_ent = state_ent * keep_mask.float()

            min_vals = q_val_.min(dim=1, keepdim=True)[0]
            max_vals = q_val_.max(dim=1, keepdim=True)[0]
            denom = max_vals - min_vals
            denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)

            # 逐行归一化 (safe)
            normalized_q_val_ = (q_val_ - min_vals) / denom

            new_sel = normalized_q_val_ * state_ent
            new_sel = new_sel.masked_fill(~keep_mask, -1e9)
            new_sel = torch.nan_to_num(new_sel, nan=-1e9, posinf=-1e9, neginf=-1e9)
            if args.dqn_action_select == 'softmax':
                temp = max(args.dqn_temp, 1e-4)
                probs = torch.softmax(new_sel / temp, dim=1)
                action = torch.multinomial(probs, 1).view(-1)
            else:
                action = new_sel.max(1)[1].view(-1)
            if len(skip_rows) > 0:
                for r in skip_rows:
                    action[r] = -1

            # Reset condition: (a) all Q in a group are negative, or (b) selected entropy << max entropy
            reset_net = False
            for i in range(q_val_.size(0)):
                if q_val_[i].max().item() < 0:
                    reset_net = True
                    break
                if action[i] >= 0:
                    ent_row = state_ent[i]
                    ent_max = ent_row.max().item()
                    sel_ent = ent_row[action[i]].item()
                    if ent_max > 0 and sel_ent < 0.5 * ent_max:
                        reset_net = True
                        break
            print('this is select by dqn')
            del (state)
        action = action.cpu()
    else:
        action = Variable(torch.Tensor([np.random.choice(range(args.rl_pool), state.size()[0], replace=True)]).type(
            torch.LongTensor).view(-1))  # .cuda()
        print('this is select by random!!!!!!!')
    
    
    return action, steps_done, ent, reset_net


def add_labeled_images(args, list_existing_images, region_candidates, train_set, action_list, budget, n_ep):
    """This function adds an image, indicated by 'action_list' out of 'region_candidates' list
     and adds it into the labeled dataset and the list of existing images.

    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param list_existing_images: (list) of tuples (image idx, region_x, region_y) of all regions that have
            been selected in the past to add them to the labeled set.
    :param region_candidates: (list) List of all possible regions to add.
    :param train_set: (torch.utils.data.Dataset) Training set.
    :param action_list: Selected indexes of the regions in 'region_candidates' to be labeled.
    :param budget: (int) Number of maximum regions we want to label.
    :param n_ep: (int) Number of episode.
    :return: List of existing images, updated with the new image.
    """

    lab_set = open(os.path.join(args.ckpt_path, args.exp_name, 'labeled_set_' + str(n_ep) + '.txt'), 'a')
    for i, action in enumerate(action_list):
        if action < 0:
            continue
        if train_set.get_num_labeled_regions() >= budget:
            print ('Budget reached with ' + str(train_set.get_num_labeled_regions()) + ' regions!')
            break
        im_toadd = region_candidates[i, action, 0]
        train_set.add_index(im_toadd, (region_candidates[i, action, 1], region_candidates[i, action, 2]))
        list_existing_images.append(tuple(region_candidates[i, action]))
        lab_set.write("%i,%i,%i" % (
            im_toadd, region_candidates[i, action, 1], region_candidates[i, action, 2]))
        lab_set.write("\n")
    print('Labeled set has now ' + str(train_set.get_num_labeled_regions()) + ' labeled regions.')

    return list_existing_images


def add_kl_pool2(state, n_cl=19):
    device = state.device
    dtype = state.dtype
    eps = 1e-8

    hist = state[:, :, :n_cl + 1]
    all_cand = hist.reshape(-1, n_cl + 1).transpose(0, 1)  # [n_cl+1, total]
    sim_matrix = torch.zeros(state.size(0), state.size(1), 32, device=device, dtype=dtype)

    for i in range(state.size(0)):
        pool_hist = hist[i]  # [pool, n_cl+1]
        numerator = pool_hist.unsqueeze(-1)
        denominator = all_cand.unsqueeze(0)
        prov_sim = (numerator * torch.log((numerator + eps) / (denominator + eps))).sum(dim=1)

        for j in range(pool_hist.size(0)):
            row = prov_sim[j]
            row_min, row_max = row.min(), row.max()
            if (row_max - row_min).abs() < 1e-6:
                hist_bins = torch.zeros(32, device=device, dtype=dtype)
                hist_bins[0] = 1.0
            else:
                hist_bins = torch.histc(row, bins=32, min=row_min.item(), max=row_max.item())
                hist_bins = hist_bins / (hist_bins.sum() + eps)
                hist_bins = hist_bins.to(dtype=dtype)
            sim_matrix[i, j] = hist_bins

    return torch.cat([state, sim_matrix], dim=2)


def create_feature_vector_3H_region_kl_sim(pred_region, ent_region, train_set, num_classes=19, reg_sz=(128, 128)):
    unique, counts = np.unique(pred_region, return_counts=True)
    sample_stats = np.zeros(num_classes + 1) + 1E-7
    sample_stats[unique.astype(int)] = counts
    sample_stats = sample_stats.tolist()
    sz = ent_region.size()
    ks_x = int(reg_sz[0] // 8)
    ks_y = int(reg_sz[1] // 8)
    with torch.no_grad():
        sample_stats += (5 - F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(
            -1)).tolist()  # min entropy
        sample_stats += F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        sample_stats += F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
    if len(train_set.balance_cl) > 0:
        inp_hist = sample_stats[0:num_classes + 1]
        sim_sample = entropy(np.repeat(np.asarray(inp_hist)[:, np.newaxis], len(train_set.balance_cl), axis=1),
                             np.asarray(train_set.balance_cl).transpose(1, 0))
        hist, _ = np.histogram(sim_sample, bins=32)
        sim_lab = list(hist / hist.sum())
        sample_stats += sim_lab
    else:
        sample_stats += [0.0] * 32
    return sample_stats


def create_feature_vector_3H_region_kl(pred_region, ent_region, num_classes=19, reg_sz=(128, 128)):
    unique, counts = np.unique(pred_region, return_counts=True)
    sample_stats = np.zeros(num_classes + 1) + 1E-7
    sample_stats[unique.astype(int)] = counts
    sample_stats = sample_stats.tolist()
    sz = ent_region.size()
    ks_x = int(reg_sz[0] // 8)
    ks_y = int(reg_sz[1] // 8)
    with torch.no_grad():
        sample_stats += (5 - F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(
            -1)).tolist()  # min entropy
        # print(len(sample_stats))
        sample_stats += F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        # print(len(sample_stats))
        sample_stats += F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        # print(len(sample_stats))
    return sample_stats


def optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, BATCH_SIZE=32, GAMMA=0.999,
                        dqn_epochs=1):
    """This function optimizes the policy network

    :(ReplayMemory) memory: Experience replay buffer
    :param Transition: definition of the experience replay tuple
    :param policy_net: Policy network
    :param target_net: Target network
    :param optimizerP: Optimizer of the policy network
    :param BATCH_SIZE: (int) Batch size to sample from the experience replay
    :param GAMMA: (float) Discount factor
    :param dqn_epochs: (int) Number of epochs to train the DQN
    """
    # Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    if len(memory) < BATCH_SIZE:
        return
    print('Optimize model...')
    print (len(memory))
    policy_net.train()
    loss_item = 0
    for ep in range(dqn_epochs):
        optimizerP.zero_grad()
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8).cuda()

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        non_final_next_states_subset = torch.cat([s for s in batch.next_state_subset
                                                  if s is not None])

        state_batch = torch.cat(batch.state)
        state_batch_subset = torch.cat(batch.state_subset)
        action_batch = torch.Tensor([batch.action]).view(-1).type(torch.LongTensor)
        reward_batch = torch.Tensor([batch.reward]).view(-1).cuda()
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        q_val = policy_net(state_batch.cuda(), state_batch_subset.cuda())
        # Log Q statistics for analysis (mean, max, min)
        q_mean = q_val.mean().item()
        q_max = q_val.max().item()
        q_min = q_val.min().item()
        q_log_path = os.path.join(args.ckpt_path, args.exp_name, 'q_values.txt')
        with open(q_log_path, 'a') as qf:
            qf.write(f"{q_mean:.6f},{q_max:.6f},{q_min:.6f}\n")
        state_batch.cpu()
        state_batch_subset.cpu()
        state_action_values = q_val.gather(1, action_batch.unsqueeze(1).cuda())
        action_batch.cpu()
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE).cuda()
        # Double dqn, so we compute the values with the target network, we choose the actions with the policy network
        if non_final_mask.sum().item() > 0:
            v_val_act = policy_net(non_final_next_states.cuda(), non_final_next_states_subset.cuda()).detach()
            v_val = target_net(non_final_next_states.cuda(), non_final_next_states_subset.cuda()).detach()
            non_final_next_states.cpu()
            non_final_next_states_subset.cpu()
            act = v_val_act.max(1)[1]
            next_state_values[non_final_mask] = v_val.gather(1, act.unsqueeze(1)).view(-1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values.view(-1), expected_state_action_values)
        cri = nn.MSELoss()
        loss = cri(state_action_values.view(-1), expected_state_action_values)
        loss_item += loss.item()
        progress_bar(ep, dqn_epochs, '[DQN loss %.5f]' % (
                loss_item / (ep + 1)))
        loss.backward()
        optimizerP.step()

        del (q_val)
        del (expected_state_action_values)
        del (loss)
        del (next_state_values)
        del (reward_batch)
        if non_final_mask.sum().item() > 0:
            del (act)
            del (v_val)
            del (v_val_act)
        del (state_action_values)
        del (state_batch)
        del (action_batch)
        del (non_final_mask)
        del (non_final_next_states)
        del (batch)
        del (transitions)
    lab_set = open(os.path.join(args.ckpt_path, args.exp_name, 'q_loss.txt'), 'a')
    lab_set.write("%f" % (loss_item))
    lab_set.write("\n")
    lab_set.close()