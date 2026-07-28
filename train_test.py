import numpy as np
import torch
import torch.nn.functional as F
from utils.metrics import multi_class_eval


def train_one_epoch(model, loss_func, optimizer, train_x_left, train_x_right, train_y,
                    fold_index, epoch, batch_size, device):
    train_index = np.arange(len(train_y))
    np.random.shuffle(train_index)
    step = 0
    model.train()

    warmup_epochs = 30
    aux_scale = min(1.0, epoch / max(warmup_epochs, 1))

    beta_align = 0.1 * aux_scale
    beta_faith = 0.05 * aux_scale
    beta_kg_align = 0.05 * aux_scale
    beta_ortho = 0.01

    tau = 0.2
    
    model.cache_kg_embedding()

    for j in range(0, len(train_index), batch_size):
        index = train_index[j:j + batch_size]
        x_left = train_x_left[index]
        x_right = train_x_right[index]
        y = train_y[index].squeeze(dim=-1)

        optimizer.zero_grad()

        y_pred, _, left_emb, right_emb = model(x_left, x_right, return_alpha=True)
        pred_loss = loss_func(y_pred, y.to(device))

        att_weights = model.hyper_mol.last_att_weights
        k = max(1, int(att_weights.shape[0] * 0.05))
        _, topk_sub = torch.topk(att_weights, k)

        with torch.no_grad():
            sub_cf_pred, _, _, _ = model(x_left, x_right, mask_substructures=topk_sub, return_alpha=False)

        faith_loss = model.decoder.get_hierarchical_faith_loss(
            left_emb, right_emb,
            pred_orig=y_pred,
            sub_cf_pred=sub_cf_pred,
            top_ratio=0.1,
            tau=tau
        )

        align_loss = model.hyper_mol.get_align_loss()

        kg_edge_emb = model.kg.gat_layers[0].edge_embedding.weight
        kg_align_loss = model.decoder.get_kg_alignment_loss(kg_edge_emb)

        ortho_loss = model.decoder.get_orthogonality_loss()

        train_loss = (pred_loss
                      + beta_align * align_loss
                      + beta_faith * faith_loss
                      + beta_kg_align * kg_align_loss
                      + beta_ortho * ortho_loss)

        y_pred_np = torch.softmax(y_pred, dim=-1).detach().cpu().numpy()
        train_acc, train_f1, train_precision, train_recall, train_kappa = multi_class_eval(
            y.numpy(),
            y_pred_np)

        print(
            "fold:{} epoch:{} step:{} "
            "train loss:{:.4f}(pred:{:.4f} aln:{:.4f} caus:{:.4f} kg_aln:{:.4f} ort:{:.4f}) "
            "acc:{:.3f} f1:{:.3f} pre:{:.3f} rec:{:.3f} kap:{:.3f}"
            .format(fold_index, epoch, step,
                    train_loss.item(), pred_loss.item(), align_loss.item(),
                    faith_loss.item(), kg_align_loss.item(), ortho_loss.item(),
                    train_acc, train_f1, train_precision, train_recall, train_kappa)
        )
       
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        model.cache_kg_embedding()  # Update cached KG embeddings after each parameter update
        step += 1


def calibrate_logits(logits, class_prior, temperature=1.0, alpha=0.3):
    log_prior = torch.log(class_prior.clamp(min=1e-8))
    log_prior = log_prior - log_prior.mean()
    calibrated = logits / temperature - alpha * log_prior.unsqueeze(0)
    return calibrated


def compute_class_prior(train_y_numpy, num_classes):
    labels = train_y_numpy.squeeze().astype(int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    prior = counts / counts.sum()
    return torch.tensor(prior, dtype=torch.float32)


def test(model, loss_func, valid_x_left, valid_x_right, valid_y,
         fold_index, epoch, batch_size, device, class_prior=None, temperature=1.0, alpha=0.3):
    model.eval()
    with torch.no_grad():
        model.cache_kg_embedding()

        valid_index = np.arange(len(valid_y))
        y = torch.LongTensor([])
        y_pred = torch.tensor([]).to(device)

        for j in range(0, len(valid_index), batch_size):
            index = valid_index[j:j + batch_size]
            x_left = valid_x_left[index]
            x_right = valid_x_right[index]
            y = torch.concat([y, valid_y[index].squeeze(dim=-1)])
            pred, _, _, _  = model(x_left, x_right, return_alpha=False)
            y_pred = torch.concat([y_pred, pred])

        valid_loss = loss_func(y_pred, y.to(device))
        if class_prior is not None:
            prior_dev = class_prior.to(device)
            y_pred_cal = calibrate_logits(y_pred, prior_dev, temperature, alpha)
        else:
            y_pred_cal = y_pred

        y_pred = torch.softmax(y_pred_cal, dim=-1).cpu().detach().numpy()
        valid_acc, valid_f1, valid_precision, valid_recall, valid_kappa = multi_class_eval(y.numpy(), y_pred)
        print(
            "fold:{} epoch:{}        "
            "valid loss:{:.6f}, valid acc:{:.3f}, valid f1:{:.3f}, valid precision:{:.3f}, valid recall:{:.3f}, valid kappa:{:.3f}"
            .format(
                fold_index, epoch,
                valid_loss.item(), valid_acc, valid_f1, valid_precision, valid_recall, valid_kappa
            ))

        score = [valid_acc, valid_f1, valid_precision, valid_recall, valid_kappa]
        
        return score
