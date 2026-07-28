import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from model.CausalDDI import FaithDDI
from utils.data_loader import load_data, get_train_test
from train_test import train_one_epoch, test, compute_class_prior
from utils.pytorchtools import EarlyStopping
from utils.logger import Logger


CONDITION_DEFAULTS = {
    'S1': dict(lr=5e-4, weight_decay=1e-4, label_smoothing=0.05,
               dropout_hyper=0.2, dropout_clf=0.2,
               dropout_coattn=0.1, dropout_kg_emb=0.1, prior_alpha=0.05),
    'S2': dict(lr=3e-4, weight_decay=2e-4, label_smoothing=0.08,
               dropout_hyper=0.3, dropout_clf=0.3,
               dropout_coattn=0.15, dropout_kg_emb=0.2, prior_alpha=0.1),
    'S3': dict(lr=2e-4, weight_decay=5e-4, label_smoothing=0.1,
               dropout_hyper=0.4, dropout_clf=0.4,
               dropout_coattn=0.2, dropout_kg_emb=0.3, prior_alpha=0.15),
}


def apply_condition_defaults(args):
    defaults = CONDITION_DEFAULTS[args.condition]
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def run(args):
    args = apply_condition_defaults(args)
    print(f"[Config] condition={args.condition}")
    print(f"lr={args.lr}, weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}")
    print(f"dropout_hyper={args.dropout_hyper}, dropout_clf={args.dropout_clf}, "
          f"dropout_coattn={args.dropout_coattn}, dropout_kg_emb={args.dropout_kg_emb}")
    print(f"prior_alpha={args.prior_alpha}")

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    data_path = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name)
    kg_g, smiles = load_data(data_path, device=device)
    train_sample, valid_sample, test_sample = get_train_test(
        data_path,
        fold_num=args.fold_num,
        condition=args.condition)

    scores = []
    for i in range(0, args.fold_num):
        print(f"============================ Fold {i} Start ============================")
        model_save_dir = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name, args.condition, f'Fold{i}')
        os.makedirs(model_save_dir, exist_ok=True)

        train_x_left = train_sample[i][:, 0]
        train_x_right = train_sample[i][:, 1]
        train_y = train_sample[i][:, 2:]

        valid_x_left = valid_sample[i][:, 0]
        valid_x_right = valid_sample[i][:, 1]
        valid_y = valid_sample[i][:, 2:]

        test_x_left = test_sample[i][:, 0]
        test_x_right = test_sample[i][:, 1]
        test_y = test_sample[i][:, 2:]

        train_y = torch.from_numpy(train_y).long()
        valid_y = torch.from_numpy(valid_y).long()
        test_y = torch.from_numpy(test_y).long()

        class_num = 65 if args.ddi_name == 'Deng' else 86
        class_prior = compute_class_prior(train_sample[i][:, 2:], class_num)
        
        model = FaithDDI(
            kg_g, smiles, args.hidden_dim, args.num_layer, class_num, 
            args.condition, args.dropout_hyper,
            args.dropout_clf, args.dropout_coattn, args.dropout_kg_emb,
        ).to(device)
        
        loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        if i == 0:
            print(model)

        weight_p, bias_bn_emb_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name or 'bn' in name or 'embedding' in name or 'gate' in name:
                bias_bn_emb_p += [p]
            else:
                weight_p += [p]
        model_parameters = [
            {'params': weight_p, 'weight_decay': args.weight_decay},
            {'params': bias_bn_emb_p, 'weight_decay': 0},
        ]
        optimizer = optim.Adam(model_parameters, lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        best_val_score = None
        for epoch in range(args.epoch):
            train_one_epoch(model, loss_func, optimizer, train_x_left, train_x_right, train_y,
                            i, epoch, args.batch_size, device)

            val_score, _ = test(model, loss_func, valid_x_left, valid_x_right, valid_y, i, epoch, 
                                args.batch_size, device, class_prior=class_prior, temperature=1.0, alpha=args.prior_alpha)

            val_f1 = val_score[1]
            val_monitor = val_f1
            if epoch > 50:
                early_stopping(val_monitor, model)
                if early_stopping.counter == 0:
                    best_val_score = val_score
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f'best_model_{args.num_layer}_{args.hidden_dim}_C.pth'))
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    break

            if best_val_score is None:
                print(best_val_score)
            else:
                val_acc, val_f1, val_precision, val_recall, val_kappa = best_val_score
                print(
                    "val acc:{:.5f}, val f1:{:.5f}, val precision:{:.5f}, val recall:{:.5f}, val kappa:{:.5f}".format(
                        val_acc, val_f1, val_precision, val_recall, val_kappa
                    ))

        print(f"\n========== Fold {i} Training Finished. Evaluating on Test Set. ==========")
        best_model_path = os.path.join(model_save_dir, f'best_model_{args.num_layer}_{args.hidden_dim}_C.pth')
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        final_test_score = test(model, loss_func, test_x_left, test_x_right, test_y, i, 0, 
                                args.batch_size, device, class_prior=class_prior, temperature=1.0, alpha=args.prior_alpha)

        scores.append(final_test_score)
        print('Test set score:', scores)

    scores = np.array(scores)
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)

    print(f"\033[1;31mFinal DDI Result ({args.condition}):\n"
            f"Acc: {mean_scores[0]:.4f} ± {std_scores[0]:.4f}\n"
            f"F1 : {mean_scores[1]:.4f} ± {std_scores[1]:.4f}\n"
            f"Pre: {mean_scores[2]:.4f} ± {std_scores[2]:.4f}\n"
            f"Rec: {mean_scores[3]:.4f} ± {std_scores[3]:.4f}\n"
            f"Kap: {mean_scores[4]:.4f} ± {std_scores[4]:.4f}\033[0m")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='')

    ap.add_argument('--batch_size', type=int, default=2 ** 12)
    ap.add_argument('--fold_num', type=int, default=5)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--num_layer', type=int, default=3)
    ap.add_argument('--epoch', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=50)
    
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--label_smoothing', type=float, default=None)
    ap.add_argument('--dropout_hyper', type=float, default=None)
    ap.add_argument('--dropout_clf', type=float, default=None)
    ap.add_argument('--dropout_coattn', type=float, default=None)
    ap.add_argument('--dropout_kg_emb', type=float, default=None)
    ap.add_argument('--prior_alpha', type=float, default=None)

    ap.add_argument('--condition', type=str, choices=['S1', 'S2', 'S3'], default='S2')
    ap.add_argument('--data_path', type=str, default='./data')
    ap.add_argument('--kg_name', type=str, default='DRKG')
    ap.add_argument('--ddi_name', type=str, choices=['Deng', 'Ryu'], default='Deng')

    args = ap.parse_args(args=[])
    print(args)

    terminal = sys.stdout
    log_file = './log/{}_{}_{}_{}.txt'. \
        format(args.ddi_name, args.condition, args.hidden_dim, args.num_layer)
    sys.stdout = Logger(log_file, terminal)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on', device)

    run(args)

