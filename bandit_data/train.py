"""
the code is based on the Karpathy's train.py for training nanoGPT - https://github.com/karpathy/nanoGPT/blob/master/train.py
"""
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from model import RegularModel, ModelConfig, TramModel, FullModel
from utils import load_ijcai_data

# -----------------------------------------------------------------------------
# I/O
out_dir = 'logs/final_submission'
eval_interval = 20
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
# wandb logging
wandb_log = False  # disabled by default
wandb_project = None #'final_submission'
wandb_run_name = None#'regular' + str(time.time())
# data
dataset = 'IJCAI15'
batch_size = 1024
features = ['age_range', 'gender', 'dow']
proxies = ['log_signal_0', 'signal_3_binary']  # we are not using add-to-cart as proxies
target = 'signal_2_binary'
action_col = 'item_id'
# model
n_embd = 148
dropout = 0.0
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-3  # max learning rate
max_iters = 1000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 50  # how many steps to warm up for
lr_decay_iters = 1000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
master_process = True
seed_offset = 1337

if master_process:
    os.makedirs(out_dir, exist_ok=True)
#torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
X_train, A_train, Z_train, Y_train = load_ijcai_data('item_id', 10, is_train=True)
X_test, A_test, Z_test, Y_test = load_ijcai_data('item_id', 10, is_train=False)
output_size = Y_train.shape[1]
input_size_x = X_train.shape[1]
input_size_a = A_train.shape[1]
input_size_z = Z_train.shape[1]
config['input_size_x'] = input_size_x
config['input_size_a'] = input_size_a
config['input_size_z'] = input_size_z
config['output_size'] = output_size


def softmax(w, t=1.0):
    """
    softmax with max normalization
    """
    max_values, _ = torch.max(w, dim=1, keepdim=True)
    e = torch.exp((w - max_values) / t)
    return e / torch.sum(e, 1).unsqueeze(-1)

def get_batch(split):
    if split == 'train':
        X, A, Z, Y = X_train, A_train, Z_train, Y_train
    else:
        X, A, Z, Y = X_test, A_test, Z_test, Y_test
    ix = torch.randint(len(X), (batch_size,))
    x = torch.stack([torch.from_numpy((X[i]).astype(np.float32)) for i in ix])
    a = torch.stack([torch.from_numpy((A[i]).astype(np.float32)) for i in ix])
    z = torch.stack([torch.from_numpy((Z[i]).astype(np.float32)) for i in ix])
    y = torch.stack([torch.from_numpy((Y[i]).astype(np.float32)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a = x.pin_memory().to(device, non_blocking=True), a.pin_memory().to(device, non_blocking=True)
        z, y = z.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, a, z, y = x.to(device), a.to(device), z.to(device), y.to(device)

    # drop proxies -3 - student from dupi (just regular model applied to teacher labels)
    # drop proxies 0 - teacher from dupi (just privileged model with all priv info)
    # drop proxies -1 - regular model
    # drop proxies -2 - tram model
    if 0 < drop_proxies <= 1:
        if split == 'train':
            mask = torch.rand(len(x)) < drop_proxies
            z[mask] = -1.
        else:
            z = (torch.ones((len(x), Z.shape[1])) * -1.0).to(device)

    return x, a, z, y

num_runs = 10
drop_proxies_values = [0., -3., -2., -1.]
valloss_values = {key: np.zeros((max_iters // eval_interval + 1, num_runs)) for key in drop_proxies_values}
trainloss_values = {key: np.zeros((max_iters // eval_interval + 1, num_runs)) for key in drop_proxies_values}
metric_values = {key: np.zeros((max_iters // eval_interval + 1, num_runs)) for key in drop_proxies_values}

for drop_proxies in drop_proxies_values:
    for run in range(num_runs):
        torch.manual_seed(run + seed_offset)
        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9
        config['drop_proxies'] = drop_proxies
        # model init
        model_args = dict(n_embd=n_embd, bias=bias, dropout=dropout)  # start with model_args from command line
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelConfig(**model_args)
        if drop_proxies == -2.0:
            model= TramModel(gptconf)
        elif drop_proxies == -1.:
            model = RegularModel(gptconf)
        # drop_proxies = -3 corresponds to student in Generalized distillation
        elif drop_proxies == -3.:
            model = RegularModel(gptconf)
        # full model uses priv info, particularly, drop_proxies = 0 corresponds to privileged model (teacher)
        else:
            model = FullModel(gptconf)
        model.to(device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        # optimizer
        optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        checkpoint = None  # free up memory

        # helps estimate an arbitrarily accurate loss over either split using many batches
        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                roc_aucs = np.zeros(eval_iters)
                for k in range(eval_iters):
                    X, A, Z, Y = get_batch(split)
                    with ctx:
                        logits, _, loss = model(X, A, Z, Y)
                    if split == 'val':
                        roc_auc = 2 * roc_auc_score(Y[:, 0].cpu().numpy(),
                                                    torch.min(logits, dim=1)[0].cpu().float().numpy()) - 1
                        roc_aucs[k] = roc_auc
                    losses[k] = loss.item()
                out[split] = losses.mean()
            X_val = torch.from_numpy((X_test).astype(np.float32)).to(device)
            A_val = torch.from_numpy((A_test).astype(np.float32)).to(device)
            Z_val = torch.from_numpy((Z_test).astype(np.float32)).to(device)
            Y_val = torch.from_numpy((Y_test).astype(np.float32)).to(device)
            # Z_dummy = torch.zeros_like(Z_val).to(device)
            logits_val, loss_val, _ = model(X_val, A_val, Z_val, Y_val)
            roc_auc = 2 * roc_auc_score(Y_val[:, 0].cpu().numpy(), torch.min(logits_val, dim=1)[0].cpu().numpy()) - 1
            print(f'Test (whole set) ROC AUC: {roc_auc}')
            print(f'Test ROC AUC: {roc_aucs.mean()}, std: {roc_aucs.std()}')
            model.train()
            return out, roc_aucs.mean()

        # learning rate decay scheduler (cosine with warmup)
        def get_lr(it):
            # 1) linear warmup for warmup_iters steps
            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)

        # logging
        if wandb_log and master_process:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)

        # training loop
        X, A, Z, Y = get_batch('train')  # fetch the very first batch
        # drop_proxies = -3 corresponds to student in Generalized distillation
        # in this case we apply softmax to teacher labels and take it as target
        if drop_proxies == -3.:
            teacher_model = FullModel(gptconf)
            teacher_ckpt_ = torch.load(os.path.join(out_dir, f'ckpt_{0}_{run}.pt'))
            teacher_model.load_state_dict(teacher_ckpt_['model'])
            teacher_model = teacher_model.to(device)
            with torch.no_grad():
                teacher_logits, _, _ = teacher_model(X, A, Z, Y)
                Y = softmax(teacher_logits)
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = model  # unwrap DDP container if needed
        while True:
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % eval_interval == 0 and master_process:
                losses, roc_auc_local = estimate_loss()
                #roc_auc = estimate_metric()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                valloss_values[drop_proxies][iter_num // eval_interval, run] = losses['val']
                trainloss_values[drop_proxies][iter_num // eval_interval, run] = losses['train']
                metric_values[drop_proxies][iter_num // eval_interval, run] = roc_auc_local
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        #"val/roc_auc_norm": roc_auc,
                        "val/roc_auc": roc_auc_local,
                    })
                if losses['val'] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses['val']
                    best_val_metric = roc_auc_local
                    best_train_loss = losses['train']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        print(f"saving checkpoint to {out_dir}")
                        torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{int(drop_proxies)}_{run}.pt'))
            if iter_num == 0 and eval_only:
                break

            with ctx:
                logits, loss, _ = model(X, A, Z, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, A, Z, Y = get_batch('train')
            # drop_proxies = -3 corresponds to student in Generalized distillation
            # in this case we apply softmax to teacher labels and take it as target
            if drop_proxies == -3.:
                with torch.no_grad():
                    teacher_logits, _, _ = teacher_model(X, A, Z, Y)
                    Y = softmax(teacher_logits)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item()
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                if wandb_log and master_process:
                    # 🐝 Close your wandb run
                    wandb.finish()
                break

print(metric_values)
print(valloss_values)
print(trainloss_values)

with open(os.path.join(out_dir, 'best_val_losses.pickle'), 'wb') as handle:
    pickle.dump(valloss_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(out_dir, 'best_train_losses.pickle'), 'wb') as handle:
    pickle.dump(trainloss_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(out_dir, 'best_metrics.pickle'), 'wb') as handle:
    pickle.dump(metric_values, handle, protocol=pickle.HIGHEST_PROTOCOL)