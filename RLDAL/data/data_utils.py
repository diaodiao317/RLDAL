import os

import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from data import tui, tui_al, kvasir, kvasir_al, tn3k, tn3k_al, acdc_al, acdc_2d


SUPPORTED_DATASETS = {'ACDC', 'TUI', 'KVASIR', 'TN3K'}

def get_data(data_path, tr_bs, vl_bs, n_workers=0, scale_size=0, input_size=(256, 512),
             supervised=False, num_each_iter=1, only_last_labeled=False, dataset='ACDC', test=False,
             al_algorithm='ralis', full_res=False,
             region_size=128):
    print('Loading data...')
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {sorted(SUPPORTED_DATASETS)}")

    candidate_set = None
    input_transform, target_transform, train_joint_transform, train_joint_transform_sueprvised,val_joint_transform, al_train_joint_transform = \
        get_transforms(scale_size, input_size, region_size, supervised, test, al_algorithm, full_res, dataset)

    # To train pre-trained segmentation network and upper bounds.
    if supervised:
        if 'gta' in dataset:
            train_set = gtav.GTAV('fine', 'train',
                                  data_path=data_path,
                                  joint_transform=train_joint_transform,
                                  transform=input_transform,
                                  target_transform=target_transform,
                                  camvid=True if dataset == 'gta_for_camvid' else False)
            val_set = gtav.GTAV('fine', 'val',
                                data_path=data_path,
                                joint_transform=val_joint_transform,
                                transform=input_transform,
                                target_transform=target_transform,
                                camvid=True if dataset == 'gta_for_camvid' else False)
        elif dataset == 'camvid':
            train_set = camvid.Camvid('fine', 'train',
                                      data_path=data_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform)
            val_set = camvid.Camvid('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'camvid_subset':
            train_set = camvid.Camvid('fine', 'train',
                                      data_path=data_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform, subset=True)
            val_set = camvid.Camvid('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'BUSI':
            # train_set = busi.BUSI('fine', 'train',
            #                           data_path=data_path,
            #                           joint_transform=train_joint_transform,
            #                           transform=input_transform,
            #                           target_transform=target_transform, subset=True)
            # val_set = busi.BUSI('fine', 'val',
            #                         data_path=data_path,
            #                         joint_transform=val_joint_transform,
            #                         transform=input_transform,
                                    # target_transform=target_transform)
            train_set = busi.BUSI('fine', 'train',
                                      data_path=data_path,
                                      joint_transform=None,
                                      transform=input_transform,
                                      target_transform=target_transform, subset=True)
            val_set = busi.BUSI('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=None,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'TUI':
            # train_set = busi.BUSI('fine', 'train',
            #                           data_path=data_path,
            #                           joint_transform=train_joint_transform,
            #                           transform=input_transform,
            #                           target_transform=target_transform, subset=True)
            # val_set = busi.BUSI('fine', 'val',
            #                         data_path=data_path,
            #                         joint_transform=val_joint_transform,
            #                         transform=input_transform,
                                    # target_transform=target_transform)
            train_set = tui.TUI('fine', 'train',
                                      data_path=data_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform, subset=True)
            val_set = tui.TUI('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=None,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'KVASIR':
            # train_set = busi.BUSI('fine', 'train',
            #                           data_path=data_path,
            #                           joint_transform=train_joint_transform,
            #                           transform=input_transform,
            #                           target_transform=target_transform, subset=True)
            # val_set = busi.BUSI('fine', 'val',
            #                         data_path=data_path,
            #                         joint_transform=val_joint_transform,
            #                         transform=input_transform,
                                    # target_transform=target_transform)
            train_set = kvasir.KVASIR('fine', 'train',
                                      data_path=data_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform, subset=True)
            val_set = kvasir.KVASIR('fine', 'test',
                                    data_path=data_path,
                                    joint_transform=None,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'ACDC':
            resize_dim = tuple(input_size) if isinstance(input_size, (list, tuple)) else (256, 256)
            acdc_root = os.path.join(data_path, 'ACDC-dataset', 'ACDC', '2d')
            train_set = acdc_al.ACDC_al('fine', 'train',
                                        data_path=data_path,
                                        joint_transform=train_joint_transform,
                                        transform=input_transform,
                                        target_transform=target_transform, num_each_iter=num_each_iter,
                                        only_last_labeled=only_last_labeled,
                                        split='train',
                                        region_size=region_size)
            candidate_set = acdc_al.ACDC_al('fine', 'train',
                                             data_path=data_path,
                                             joint_transform=None,
                                             candidates_option=True,
                                             transform=input_transform,
                                             target_transform=target_transform,
                                             split='train',
                                             region_size=region_size)
            train_set_final = acdc_al.ACDC_al('fine', 'train',
                                              data_path=data_path,
                                              joint_transform=train_joint_transform,
                                              transform=input_transform,
                                              target_transform=target_transform, num_each_iter=num_each_iter,
                                              only_last_labeled=only_last_labeled,
                                              split='select_unlab_region',
                                              region_size=region_size)
            candidate_set_final = acdc_al.ACDC_al('fine', 'train',
                                                   data_path=data_path,
                                                   joint_transform=None,
                                                   candidates_option=True,
                                                   transform=input_transform,
                                                   target_transform=target_transform,
                                                   split='select_unlab_region',
                                                   region_size=region_size)
            unlab_set_final = acdc_al.ACDC_al('fine', 'train',
                                               data_path=data_path,
                                               joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                               transform=input_transform,
                                               pretrain=True,
                                               target_transform=target_transform, num_each_iter=num_each_iter,
                                               split='unlab_final',
                                               region_size=region_size, re_all_length=True)
            full_super_set_final = acdc_al.ACDC_al('fine', 'train',
                                                   data_path=data_path,
                                                   joint_transform=train_joint_transform_sueprvised,
                                                   transform=input_transform,
                                                   pretrain=True,
                                                   target_transform=target_transform, num_each_iter=num_each_iter,
                                                   split='full_sup_final',
                                                   region_size=region_size, re_all_length=True)
            pretrain_set = acdc_al.ACDC_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain=True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='pretrain',
                                            region_size=region_size, re_all_length=True)
            unlab_set = acdc_al.ACDC_al('fine', 'train',
                                         data_path=data_path,
                                         joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                         transform=input_transform,
                                         pretrain=True,
                                         target_transform=target_transform, num_each_iter=num_each_iter,
                                         split='unlab',
                                         region_size=region_size, re_all_length=True)
            val_set = acdc_2d.ACDCSliceDataset(acdc_root, 'val', resize_dim, augment=False)
        
        elif dataset == 'ACDC':
            resize_dim = tuple(input_size) if isinstance(input_size, (list, tuple)) else (256, 256)
            acdc_root = os.path.join(data_path, 'ACDC-dataset', 'ACDC', '2d')
            train_set = acdc_al.ACDC_al('fine', 'train',
                                        data_path=data_path,
                                        joint_transform=train_joint_transform,
                                        transform=input_transform,
                                        target_transform=target_transform, num_each_iter=num_each_iter,
                                        only_last_labeled=only_last_labeled,
                                        split='train',
                                        region_size=region_size)
            candidate_set = acdc_al.ACDC_al('fine', 'train',
                                             data_path=data_path,
                                             joint_transform=None,
                                             candidates_option=True,
                                             transform=input_transform,
                                             target_transform=target_transform,
                                             split='train',
                                             region_size=region_size)
            train_set_final = acdc_al.ACDC_al('fine', 'train',
                                              data_path=data_path,
                                              joint_transform=train_joint_transform,
                                              transform=input_transform,
                                              target_transform=target_transform, num_each_iter=num_each_iter,
                                              only_last_labeled=only_last_labeled,
                                              split='select_unlab_region',
                                              region_size=region_size)
            candidate_set_final = acdc_al.ACDC_al('fine', 'train',
                                                   data_path=data_path,
                                                   joint_transform=None,
                                                   candidates_option=True,
                                                   transform=input_transform,
                                                   target_transform=target_transform,
                                                   split='select_unlab_region',
                                                   region_size=region_size)
            unlab_set_final = acdc_al.ACDC_al('fine', 'train',
                                               data_path=data_path,
                                               joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                               transform=input_transform,
                                               pretrain=True,
                                               target_transform=target_transform, num_each_iter=num_each_iter,
                                               split='unlab_final',
                                               region_size=region_size, re_all_length=True)
            full_super_set_final = acdc_al.ACDC_al('fine', 'train',
                                                   data_path=data_path,
                                                   joint_transform=train_joint_transform_sueprvised,
                                                   transform=input_transform,
                                                   pretrain=True,
                                                   target_transform=target_transform, num_each_iter=num_each_iter,
                                                   split='full_sup_final',
                                                   region_size=region_size, re_all_length=True)
            pretrain_set = acdc_al.ACDC_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain=True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='pretrain',
                                            region_size=region_size, re_all_length=True)
            unlab_set = acdc_al.ACDC_al('fine', 'train',
                                         data_path=data_path,
                                         joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                         transform=input_transform,
                                         pretrain=True,
                                         target_transform=target_transform, num_each_iter=num_each_iter,
                                         split='unlab',
                                         region_size=region_size, re_all_length=True)
            val_set = acdc_2d.ACDCSliceDataset(acdc_root, 'val', resize_dim, augment=False)
            print('len(val_set)',len(val_set))
        elif dataset == 'LA':
            resize_dim = tuple(input_size) if isinstance(input_size, (list, tuple)) else (224, 224)
            train_set = la.LA2D('fine', 'train',
                                data_path=data_path,
                                joint_transform=None,
                                transform=input_transform,
                                target_transform=target_transform,
                                resize_to=resize_dim)
            val_set = la.LA2D('fine', 'val',
                              data_path=data_path,
                              joint_transform=None,
                              transform=input_transform,
                              target_transform=target_transform,
                              resize_to=resize_dim)
        elif dataset == 'cs_upper_bound':
            train_set = cityscapes_al_splits.CityScapes_al_splits('fine', 'train',
                                                                  data_path=data_path,
                                                                  joint_transform=train_joint_transform,
                                                                  transform=input_transform,
                                                                  target_transform=target_transform, supervised=True)
            val_set = cityscapes.CityScapes('fine', 'val',
                                            data_path=data_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)

        elif dataset == 'cityscapes_subset':
            train_set = cityscapes_al_splits.CityScapes_al_splits('fine', 'train',
                                                                  data_path=data_path,
                                                                  joint_transform=train_joint_transform,
                                                                  transform=input_transform,
                                                                  target_transform=target_transform, subset=True)
            val_set = cityscapes.CityScapes('fine', 'val',
                                            data_path=data_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)

        else:
            train_set = cityscapes.CityScapes('fine', 'train',
                                              data_path=data_path,
                                              joint_transform=train_joint_transform,
                                              transform=input_transform,
                                              target_transform=target_transform)
            val_set = cityscapes.CityScapes('fine', 'val',
                                            data_path=data_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)
    # To train AL methods
    else:

        if dataset == 'cityscapes':
            if al_algorithm == 'ralis' and not test:
                split = 'train'
            else:
                split = 'test'
            train_set = cityscapes_al.CityScapes_al('fine', 'train',
                                                    data_path=data_path,
                                                    joint_transform=train_joint_transform,
                                                    joint_transform_al=al_train_joint_transform,
                                                    transform=input_transform,
                                                    target_transform=target_transform, num_each_iter=num_each_iter,
                                                    only_last_labeled=only_last_labeled,
                                                    split=split, region_size=region_size)
            candidate_set = cityscapes_al.CityScapes_al('fine', 'train',
                                                        data_path=data_path,
                                                        joint_transform=None,
                                                        candidates_option=True,
                                                        transform=input_transform,
                                                        target_transform=target_transform, split=split,
                                                        region_size=region_size)

            val_set = cityscapes_al_splits.CityScapes_al_splits('fine', 'train',
                                                                data_path=data_path,
                                                                joint_transform=val_joint_transform,
                                                                transform=input_transform,
                                                                target_transform=target_transform)

        elif dataset == 'camvid':
            train_set = camvid_al.Camvid_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='train' if al_algorithm == 'ralis' and not test else 'test',
                                            region_size=region_size)
            candidate_set = camvid_al.Camvid_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='train' if al_algorithm == 'ralis' and not test else 'test',
                                                region_size=region_size)
            pretrain_set = camvid_al.Camvid_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='pretrain' if al_algorithm == 'ralis' and not test else 'test',
                                            region_size=region_size,re_all_length=True)
            
            unlab_set = camvid_al.Camvid_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='unlab' if al_algorithm == 'ralis' and not test else 'test',
                                            region_size=region_size,re_all_length = True)
            

            val_set = camvid.Camvid('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'BUSI':
            train_set = busi_al.BUSI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='train',
                                            region_size=region_size)
            candidate_set = busi_al.BUSI_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='train',
                                                region_size=region_size)   
            train_set_final = busi_al.BUSI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='select_unlab_region',
                                            region_size=region_size)         
            candidate_set_final = busi_al.BUSI_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='select_unlab_region',
                                                region_size=region_size)
            
            unlab_set_final = busi_al.BUSI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='unlab_final',
                                            region_size=region_size,re_all_length = True)
            full_super_set_final = busi_al.BUSI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='full_sup_final',
                                            region_size=region_size,re_all_length=True)
        
            pretrain_set = busi_al.BUSI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='pretrain',
                                            region_size=region_size,re_all_length=True)
            
            unlab_set = busi_al.BUSI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='unlab',
                                            region_size=region_size,re_all_length = True)

            val_set = busi.BUSI('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=None,
                                    transform=input_transform,
                                    target_transform=target_transform)
            # val_set = busi.BUSI('fine', 'val',
            #                         data_path=data_path,
            #                         joint_transform=None,
            #                         transform=None,
            #                         target_transform=target_transform)
        elif dataset == 'TUI':
            train_set = tui_al.TUI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='train',
                                            region_size=region_size)
            candidate_set = tui_al.TUI_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='train',
                                                region_size=region_size)   
            train_set_final = tui_al.TUI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='select_unlab_region',
                                            region_size=region_size)         
            candidate_set_final = tui_al.TUI_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='select_unlab_region',
                                                region_size=region_size)
            
            unlab_set_final = tui_al.TUI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='unlab_final',
                                            region_size=region_size,re_all_length = True)
            full_super_set_final = tui_al.TUI_al('fine', 'train',
                                            data_path=data_path,
            #                                 joint_transform=joint_transforms.Compose([
            #     joint_transforms.RandomCrop(input_size),
            #     joint_transforms.RandomHorizontallyFlip()
            # ]),
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='full_sup_final',
                                            region_size=region_size,re_all_length=True)
        
            pretrain_set = tui_al.TUI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='pretrain',
                                            region_size=region_size,re_all_length=True)
            
            unlab_set = tui_al.TUI_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='unlab',
                                            region_size=region_size,re_all_length = True)

            val_set = tui.TUI('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'KVASIR':
            train_set = kvasir_al.KVASIR_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='train',
                                            region_size=region_size)
            candidate_set = kvasir_al.KVASIR_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='train',
                                                region_size=region_size)   
            train_set_final = kvasir_al.KVASIR_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='select_unlab_region',
                                            region_size=region_size)         
            candidate_set_final = kvasir_al.KVASIR_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='select_unlab_region',
                                                region_size=region_size)
            
            unlab_set_final = kvasir_al.KVASIR_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='unlab_final',
                                            region_size=region_size,re_all_length = True)
            full_super_set_final = kvasir_al.KVASIR_al('fine', 'train',
                                            data_path=data_path,
            #                                 joint_transform=joint_transforms.Compose([
            #     joint_transforms.RandomCrop(input_size),
            #     joint_transforms.RandomHorizontallyFlip()
            # ]),
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='full_sup_final',
                                            region_size=region_size,re_all_length=True)
        
            pretrain_set = kvasir_al.KVASIR_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='pretrain',
                                            region_size=region_size,re_all_length=True)
            
            unlab_set = kvasir_al.KVASIR_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='unlab',
                                            region_size=region_size,re_all_length = True)

            val_set = kvasir.KVASIR('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'ACDC':
            resize_dim = tuple(input_size) if isinstance(input_size, (list, tuple)) else (256, 256)
            acdc_root = os.path.join(data_path, 'ACDC-dataset', 'ACDC', '2d')
            train_set = acdc_al.ACDC_al('fine', 'train',
                                        data_path=data_path,
                                        joint_transform=train_joint_transform,
                                        transform=input_transform,
                                        target_transform=target_transform, num_each_iter=num_each_iter,
                                        only_last_labeled=only_last_labeled,
                                        split='train',
                                        region_size=region_size)
            candidate_set = acdc_al.ACDC_al('fine', 'train',
                                             data_path=data_path,
                                             joint_transform=None,
                                             candidates_option=True,
                                             transform=input_transform,
                                             target_transform=target_transform,
                                             split='train',
                                             region_size=region_size)
            train_set_final = acdc_al.ACDC_al('fine', 'train',
                                              data_path=data_path,
                                              joint_transform=train_joint_transform,
                                              transform=input_transform,
                                              target_transform=target_transform, num_each_iter=num_each_iter,
                                              only_last_labeled=only_last_labeled,
                                              split='select_unlab_region',
                                              region_size=region_size)
            candidate_set_final = acdc_al.ACDC_al('fine', 'train',
                                                   data_path=data_path,
                                                   joint_transform=None,
                                                   candidates_option=True,
                                                   transform=input_transform,
                                                   target_transform=target_transform,
                                                   split='select_unlab_region',
                                                   region_size=region_size)
            unlab_set_final = acdc_al.ACDC_al('fine', 'train',
                                               data_path=data_path,
                                               joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                               transform=input_transform,
                                               pretrain=True,
                                               target_transform=target_transform, num_each_iter=num_each_iter,
                                               split='unlab_final',
                                               region_size=region_size, re_all_length=True)
            full_super_set_final = acdc_al.ACDC_al('fine', 'train',
                                                   data_path=data_path,
                                                   joint_transform=train_joint_transform_sueprvised,
                                                   transform=input_transform,
                                                   pretrain=True,
                                                   target_transform=target_transform, num_each_iter=num_each_iter,
                                                   split='full_sup_final',
                                                   region_size=region_size, re_all_length=True)
            pretrain_set = acdc_al.ACDC_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain=True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='pretrain',
                                            region_size=region_size, re_all_length=True)
            unlab_set = acdc_al.ACDC_al('fine', 'train',
                                         data_path=data_path,
                                         joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                         transform=input_transform,
                                         pretrain=True,
                                         target_transform=target_transform, num_each_iter=num_each_iter,
                                         split='unlab',
                                         region_size=region_size, re_all_length=True)

            val_set = acdc_2d.ACDCSliceDataset(acdc_root, 'val', resize_dim, augment=False)
            
            
        elif dataset == 'TN3K':
            print('11111getinggetinggeting')
            train_set = tn3k_al.TN3K_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='train',
                                            region_size=region_size)
            candidate_set = tn3k_al.TN3K_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='train',
                                                region_size=region_size)   
            train_set_final = tn3k_al.TN3K_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='select_unlab_region',
                                            region_size=region_size)         
            candidate_set_final = tn3k_al.TN3K_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='select_unlab_region',
                                                region_size=region_size)
            
            unlab_set_final = tn3k_al.TN3K_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='unlab_final',
                                            region_size=region_size,re_all_length = True)
            full_super_set_final = tn3k_al.TN3K_al('fine', 'train',
                                            data_path=data_path,
            #                                 joint_transform=joint_transforms.Compose([
            #     joint_transforms.RandomCrop(input_size),
            #     joint_transforms.RandomHorizontallyFlip()
            # ]),
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='full_sup_final',
                                            region_size=region_size,re_all_length=True)
        
            pretrain_set = tn3k_al.TN3K_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform_sueprvised,
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='pretrain',
                                            region_size=region_size,re_all_length=True)
            print('11111getinggetinggeting')
            unlab_set = tn3k_al.TN3K_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ]),
                                            transform=input_transform,
                                            pretrain = True,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            split='unlab',
                                            region_size=region_size,re_all_length = True)
            
            val_set = tn3k.TN3K('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
            # print('len(val_set)',len(val_set))
    if supervised:
        train_loader = DataLoader(train_set,
                              batch_size=tr_bs,
                              num_workers=n_workers, shuffle=True,
                              drop_last=False)
        val_loader = DataLoader(val_set,
                            batch_size=vl_bs,
                            num_workers=n_workers, shuffle=False)
        # print('len(val_set)',len(val_set))
        return train_loader,0,val_loader,0
    # print('-------------')
    # print(len(unlab_set),len(pretrain_set))
    print(len(unlab_set),len(unlab_set_final),len(pretrain_set),len(full_super_set_final),len(train_set),len(train_set_final),len(val_set))
    unlab_set_loader = DataLoader(unlab_set,
                                 batch_size=tr_bs,
                                 sampler=sampler.RandomSampler(data_source=unlab_set,
                                                               replacement=True,num_samples=len(unlab_set)),
                                 num_workers=10,
                                 drop_last=True)
    unlab_set_loader_final = DataLoader(unlab_set_final,
                                 batch_size=tr_bs,
                                 sampler=sampler.RandomSampler(data_source=unlab_set_final,
                                                               replacement=True,num_samples=len(unlab_set_final)),
                                 num_workers=10,
                                 drop_last=True)
    
    pretrain_loader = DataLoader(pretrain_set,
                                 batch_size=tr_bs,
                                 sampler=sampler.RandomSampler(data_source=pretrain_set,
                                                               replacement=True,num_samples=len(unlab_set)),
                                 num_workers=n_workers,
                                 drop_last=True)
    full_sup_loader = DataLoader(full_super_set_final,
                                 batch_size=tr_bs,
                                 sampler=sampler.RandomSampler(data_source=full_super_set_final,
                                                               replacement=True,num_samples=len(unlab_set_final)),
                                 num_workers=n_workers,
                                 drop_last=True)
    train_loader = DataLoader(train_set,
                              batch_size=tr_bs,
                              num_workers=n_workers, shuffle=True,
                              drop_last=True)
    train_loader_final =  DataLoader(train_set_final,
                              batch_size=tr_bs,
                              sampler=sampler.RandomSampler(data_source=train_set_final,
                                                               replacement=True,num_samples=len(unlab_set_final)),
                              num_workers=n_workers, 
                              drop_last=True)
    print('len(val_set)',len(val_set))
    val_loader = DataLoader(val_set,
                            batch_size=vl_bs,
                            num_workers=n_workers, shuffle=False)

    return pretrain_loader,pretrain_set, train_loader, train_set, val_loader, candidate_set, unlab_set_loader,unlab_set, train_set_final,train_loader_final,candidate_set_final,full_sup_loader,unlab_set_final ,unlab_set_loader_final

def get_transforms(scale_size, input_size, region_size, supervised, test, al_algorithm, full_res, dataset):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if scale_size == 0:
        print('(Data loading) Not scaling the data')
        print('(Data loading) Random crops of ' + str(input_size) + ' in training')
        print('(Data loading) No crops in validation')
        if supervised:
            train_joint_transform = joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
        else:
            train_joint_transform = joint_transforms.ComposeRegion([
                joint_transforms.RandomCropRegion(input_size, region_size=region_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
        if (not test and al_algorithm == 'ralis') and not full_res:
            val_joint_transform = joint_transforms.Scale(1024)
        else:
            val_joint_transform = None
        al_train_joint_transform = joint_transforms.ComposeRegion([
            joint_transforms.CropRegion(region_size, region_size=region_size),
            joint_transforms.RandomHorizontallyFlip()
        ])
    else:
        print('(Data loading) Scaling training data: ' + str(
            scale_size) + ' width dimension')
        print('(Data loading) Random crops of ' + str(
            input_size) + ' in training')
        print('(Data loading) No crops nor scale_size in validation')

        
        train_joint_transform = joint_transforms.ComposeRegion([
                joint_transforms.Scale(scale_size),
                joint_transforms.RandomCropRegion(input_size, region_size=region_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
        al_train_joint_transform = joint_transforms.ComposeRegion([
            joint_transforms.Scale(scale_size),
            joint_transforms.CropRegion(region_size, region_size=region_size),
            joint_transforms.RandomHorizontallyFlip()
        ])
        if dataset == 'gta_for_camvid':
            val_joint_transform = joint_transforms.ComposeRegion([
                joint_transforms.Scale(scale_size)])
        else:
            val_joint_transform = None
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    train_joint_transform_supervised = joint_transforms.Compose([
                # joint_transforms.Scale(scale_size),
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
    return input_transform, target_transform, train_joint_transform,train_joint_transform_supervised, val_joint_transform, al_train_joint_transform
