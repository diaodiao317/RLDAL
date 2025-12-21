"""Entry point to run RLDAL on ACDC with a cleaner interface.

This is a thin wrapper over the existing training code, scoped to ACDC.
"""
import argparse

from RLDAL.config import RLDALConfig
from RLDAL.trainer import RLDALTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLDAL (ACDC-focused) with defaults from utils/parser.py")

    # Paths / experiment naming
    parser.add_argument("--ckpt-path", type=str, default=RLDALConfig.ckpt_path)
    parser.add_argument("--data-path", type=str, default=RLDALConfig.data_path)
    parser.add_argument("--exp-name", type=str, default=RLDALConfig.exp_name)
    parser.add_argument("--exp-name-toload", type=str, default=RLDALConfig.exp_name_toload)
    parser.add_argument("--exp-name-toload-rl", type=str, default=RLDALConfig.exp_name_toload_rl)
    parser.add_argument("--snapshot", type=str, default=RLDALConfig.snapshot)

    # General
    parser.add_argument("--seed", type=int, default=RLDALConfig.seed)
    parser.add_argument("--dataset", type=str, default=RLDALConfig.dataset,
                        choices=['camvid', 'camvid_subset', 'cityscapes', 'cityscapes_subset', 'cs_upper_bound',
                                 'gta', 'gta_for_camvid', 'BUSI', 'TUI', 'KVASIR', 'TN3K', 'LA', 'ACDC'])
    parser.add_argument("--al-algorithm", type=str, default=RLDALConfig.al_algorithm,
                        choices=['random', 'entropy', 'bald', 'ralis'])
    parser.add_argument("--region-size", nargs='+', type=int, default=RLDALConfig.region_size)
    parser.add_argument("--input-size", nargs='+', type=int, default=RLDALConfig.input_size)
    parser.add_argument("--scale-size", type=int, default=RLDALConfig.scale_size)
    parser.add_argument("--full-res", action='store_true', default=RLDALConfig.full_res)
    parser.add_argument("--only-last-labeled", action='store_true', default=RLDALConfig.only_last_labeled)

    # Active learning
    parser.add_argument("--num-each-iter", type=int, default=RLDALConfig.num_each_iter)
    parser.add_argument("--rl-pool", type=int, default=RLDALConfig.rl_pool)
    parser.add_argument("--budget-labels", type=int, default=RLDALConfig.budget_labels)
    parser.add_argument("--rl-episodes", type=int, default=RLDALConfig.rl_episodes)
    parser.add_argument("--rl-buffer", type=int, default=RLDALConfig.rl_buffer)
    parser.add_argument("--dqn-bs", type=int, default=RLDALConfig.dqn_bs)
    parser.add_argument("--dqn-gamma", type=float, default=RLDALConfig.dqn_gamma)
    parser.add_argument("--dqn-epochs", type=int, default=RLDALConfig.dqn_epochs)
    parser.add_argument("--dqn-action-select", type=str, default=RLDALConfig.dqn_action_select,
                        choices=['epsilon', 'softmax'])
    parser.add_argument("--dqn-temp", type=float, default=RLDALConfig.dqn_temp)
    parser.add_argument("--bald-iter", type=int, default=RLDALConfig.bald_iter)

    # Optim
    parser.add_argument("--optimizer", type=str, default=RLDALConfig.optimizer,
                        choices=['SGD', 'Adam', 'RMSprop'])
    parser.add_argument("--train-batch-size", type=int, default=RLDALConfig.train_batch_size)
    parser.add_argument("--val-batch-size", type=int, default=RLDALConfig.val_batch_size)
    parser.add_argument("--epoch-num", type=int, default=RLDALConfig.epoch_num)
    parser.add_argument("--lr", type=float, default=RLDALConfig.lr)
    parser.add_argument("--lr-dqn", type=float, default=RLDALConfig.lr_dqn)
    parser.add_argument("--gamma", type=float, default=RLDALConfig.gamma)
    parser.add_argument("--gamma-scheduler-dqn", type=float, default=RLDALConfig.gamma_scheduler_dqn)
    parser.add_argument("--weight-decay", type=float, default=RLDALConfig.weight_decay)
    parser.add_argument("--momentum", type=float, default=RLDALConfig.momentum)
    parser.add_argument("--patience", type=int, default=RLDALConfig.patience)

    # Checkpoint / test flags
    parser.add_argument("--checkpointer", action='store_true', default=RLDALConfig.checkpointer)
    parser.add_argument("--load-weights", action='store_true', default=RLDALConfig.load_weights)
    parser.add_argument("--load-opt", action='store_true', default=RLDALConfig.load_opt)
    parser.add_argument("--test", action='store_true', default=RLDALConfig.test)
    parser.add_argument("--final-test", action='store_true', default=RLDALConfig.final_test)
    parser.add_argument("--final-sup-only", action='store_true', default=RLDALConfig.final_sup_only)

    # Semi-supervised knobs (kept for parity)
    parser.add_argument("--labeled-fraction", type=float, default=RLDALConfig.labeled_fraction)
    parser.add_argument("--consistency-weight", type=float, default=RLDALConfig.consistency_weight)
    parser.add_argument("--confidence-threshold", type=float, default=RLDALConfig.confidence_threshold)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RLDALConfig(
        data_path=args.data_path,
        ckpt_path=args.ckpt_path,
        exp_name=args.exp_name,
        exp_name_toload=args.exp_name_toload,
        exp_name_toload_rl=args.exp_name_toload_rl,
        snapshot=args.snapshot,
        seed=args.seed,
        dataset=args.dataset,
        al_algorithm=args.al_algorithm,
        region_size=tuple(args.region_size),
        input_size=tuple(args.input_size),
        scale_size=args.scale_size,
        full_res=args.full_res,
        only_last_labeled=args.only_last_labeled,
        num_each_iter=args.num_each_iter,
        rl_pool=args.rl_pool,
        budget_labels=args.budget_labels,
        rl_episodes=args.rl_episodes,
        rl_buffer=args.rl_buffer,
        dqn_bs=args.dqn_bs,
        dqn_gamma=args.dqn_gamma,
        dqn_epochs=args.dqn_epochs,
        dqn_action_select=args.dqn_action_select,
        dqn_temp=args.dqn_temp,
        bald_iter=args.bald_iter,
        optimizer=args.optimizer,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epoch_num=args.epoch_num,
        lr=args.lr,
        lr_dqn=args.lr_dqn,
        gamma=args.gamma,
        gamma_scheduler_dqn=args.gamma_scheduler_dqn,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        patience=args.patience,
        checkpointer=args.checkpointer,
        load_weights=args.load_weights,
        load_opt=args.load_opt,
        test=args.test,
        final_test=args.final_test,
        final_sup_only=args.final_sup_only,
        labeled_fraction=args.labeled_fraction,
        consistency_weight=args.consistency_weight,
        confidence_threshold=args.confidence_threshold,
    )

    trainer = RLDALTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
