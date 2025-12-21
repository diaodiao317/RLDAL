import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.model_utils import (
    EMA,
    add_labeled_images,
    compute_state,
    create_models,
    get_region_candidates,
    optimize_model_conv,
    select_action,
)
from utils.final_utils import (
    check_mkdir,
    create_and_load_optimizers,
    get_logfile,
    train,
    validate,
)
from utils.replay_buffer import ReplayMemory, Transition

from .config import RLDALConfig
from .data_acdc import build_acdc_loaders


class RLDALTrainer:
    """Minimal RLDAL pipeline focused on ACDC."""

    def __init__(self, cfg: RLDALConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(cfg.seed)
        check_mkdir(cfg.ckpt_path)
        check_mkdir(os.path.join(cfg.ckpt_path, cfg.exp_name))

        # Models and optimizers
        self.net, self.policy_net, self.target_net = create_models(
            dataset=cfg.dataset, al_algorithm=cfg.al_algorithm, region_size=cfg.region_size
        )
        self.optimizer, self.optimizerP = create_and_load_optimizers(
            net=self.net,
            opt_choice=cfg.optimizer,
            lr=cfg.lr,
            wd=cfg.weight_decay,
            momentum=cfg.momentum,
            ckpt_path=cfg.ckpt_path,
            exp_name_toload=cfg.exp_name_toload,
            exp_name=cfg.exp_name,
            snapshot=cfg.snapshot,
            checkpointer=cfg.checkpointer,
            load_opt=cfg.load_opt,
            policy_net=self.policy_net,
            lr_dqn=cfg.lr_dqn,
            al_algorithm=cfg.al_algorithm,
        )

        loaders = build_acdc_loaders(cfg)
        (
            self.pretrain_loader,
            self.pretrain_set,
            self.train_loader,
            self.train_set,
            self.val_loader,
            self.candidate_set,
            self.unlab_loader,
            self.unlab_set,
            self.train_set_final,
            self.train_loader_final,
            self.candidate_set_final,
            self.full_sup_loader,
            self.unlab_set_final,
            self.unlab_loader_final,
        ) = loaders

        # Losses
        self.cri_reduced = nn.CrossEntropyLoss(ignore_index=self.train_loader.dataset.ignore_label).to(self.device)
        self.cri = nn.CrossEntropyLoss(
            ignore_index=self.train_loader.dataset.ignore_label, reduction="none"
        ).to(self.device)

    # ------------------------------ helpers ------------------------------
    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # ------------------------------ stages ------------------------------
    def pretrain(self) -> None:
        cfg = self.cfg
        ema = EMA(model=self.net, alpha=0.99)
        ema.model = ema.model.to(self.device)
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum, nesterov=True
        )
        scheduler = ExponentialLR(optimizer, gamma=cfg.gamma)
        logger, best_record_pre, _ = get_logfile(
            cfg.ckpt_path, cfg.exp_name, cfg.checkpointer, cfg.snapshot, log_name="pretrain_log.txt",
            num_classes=self.train_set.num_classes,
        )

        self.net.train()
        for epoch in range(cfg.epoch_num):
            print(f"[Pretrain] Epoch {epoch + 1}/{cfg.epoch_num}")
            tr_loss, _, _, _, _ = train(
                self.pretrain_loader, self.net, self.cri, self.cri_reduced, optimizer, supervised=True
            )
            scheduler.step()
            vl_loss, val_acc, val_iu, iu_xclass, best_record_pre = validate(
                self.val_loader, self.net, self.cri_reduced, optimizer, epoch, best_record_pre, cfg
            )
            print(f"[Pretrain] loss={tr_loss:.4f} val_iu={val_iu:.4f}")

        pre_ckpt = os.path.join(cfg.ckpt_path, cfg.exp_name, "pre_train.pth")
        torch.save(self.net.cpu().state_dict(), pre_ckpt)
        self.net.to(self.device)
        ema.model = ema.model.to(self.device)
        print(f"Saved pretrain checkpoint to {pre_ckpt}")

    def active_learning(self) -> None:
        cfg = self.cfg
        # Sync target net
        if self.target_net is not None and self.policy_net is not None:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        memory = ReplayMemory(cfg.rl_buffer)
        steps_done = 0

        for n_ep in range(cfg.rl_episodes):
            print(f"=== Episode {n_ep + 1}/{cfg.rl_episodes} ===")
            logger, best_record, _ = get_logfile(
                cfg.ckpt_path, cfg.exp_name, cfg.checkpointer, cfg.snapshot,
                log_name=f"ep{n_ep}_log.txt", num_classes=self.train_set.num_classes,
            )

            budget_reached = False
            list_existing_images = []
            train_set = self.train_set
            candidate_set = self.candidate_set
            net = self.net
            policy_net = self.policy_net
            target_net = self.target_net
            optimizer = self.optimizer
            optimizerP = self.optimizerP

            # Initial candidate pool
            num_regions = cfg.num_each_iter * cfg.rl_pool
            candidates = train_set.get_candidates(num_regions_unlab=num_regions)
            candidate_set.reset()
            candidate_set.add_index(list(candidates))
            region_candidates = get_region_candidates(candidates, train_set, num_regions=num_regions)
            current_state, region_candidates, state_ent = compute_state(
                cfg, net, region_candidates, candidate_set, train_set,
                num_groups=cfg.num_each_iter, reg_sz=cfg.region_size,
            )

            while train_set.get_num_labeled_regions() < cfg.budget_labels and not budget_reached:
                action, steps_done, chosen_stats, reset_net = select_action(
                    cfg, policy_net, current_state, state_ent, steps_done
                )
                list_existing_images = add_labeled_images(
                    cfg, list_existing_images, region_candidates, train_set, action, cfg.budget_labels, n_ep
                )

                # Rebuild loader to reflect new labeled set
                train_loader = DataLoader(
                    train_set,
                    batch_size=cfg.train_batch_size,
                    num_workers=cfg.n_workers,
                    shuffle=True,
                    drop_last=False,
                )
                tr_loss, _, _, _, loss_bs = train(
                    train_loader, net, self.cri, self.cri_reduced, optimizer, supervised=False
                )

                # Reward: per-batch loss centered to encourage harder examples
                reward_slice = loss_bs[: cfg.num_each_iter] if len(loss_bs) >= cfg.num_each_iter else loss_bs
                reward_tensor = torch.tensor(reward_slice, device=self.device, dtype=torch.float32)
                if reward_tensor.numel() < cfg.num_each_iter:
                    pad = cfg.num_each_iter - reward_tensor.numel()
                    reward_tensor = torch.cat([reward_tensor, reward_tensor.new_zeros(pad)])
                reward_tensor = reward_tensor - reward_tensor.mean()

                # Next state
                candidates = train_set.get_candidates(num_regions_unlab=num_regions)
                candidate_set.reset()
                candidate_set.add_index(list(candidates))
                region_candidates = get_region_candidates(candidates, train_set, num_regions=num_regions)
                next_state, region_candidates, state_ent = compute_state(
                    cfg, net, region_candidates, candidate_set, train_set,
                    num_groups=cfg.num_each_iter, reg_sz=cfg.region_size,
                )

                memory.push(current_state, action, next_state, reward_tensor)
                current_state = next_state

                optimize_model_conv(
                    cfg,
                    memory,
                    Transition,
                    policy_net,
                    target_net,
                    optimizerP,
                    BATCH_SIZE=cfg.dqn_bs,
                    GAMMA=cfg.dqn_gamma,
                    dqn_epochs=cfg.dqn_epochs,
                )

                # Update target network periodically
                if steps_done % 5 == 0 and target_net is not None:
                    target_net.load_state_dict(policy_net.state_dict())

                if train_set.get_num_labeled_regions() >= cfg.budget_labels:
                    budget_reached = True

            logger.close()

    def run(self) -> None:
        self.pretrain()
        self.active_learning()


__all__ = ["RLDALTrainer"]
