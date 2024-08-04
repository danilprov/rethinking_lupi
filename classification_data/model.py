from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mlp(self.ln(x))
        return x


@dataclass
class ModelConfig:
    input_size_x: int = 16
    input_size_z: int = 54
    output_size: int = 2
    n_embd: int = 198
    dropout: float = 0.0
    temperature: float = 1.0
    bias: bool = True # True: bias in Linears and LayerNorms


class RegularModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_in1 = nn.Linear(config.input_size_x, config.n_embd)
        self.wpe = nn.Linear(config.n_embd, config.n_embd)
        self.block = Block(config)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.output_size, bias=False)
        self.apply(_init_weights)

    def forward(self, x, z, targets):
        x = self.lm_in1(x)
        x = self.wpe(x)
        x = self.block(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets, ignore_index=-1)

        return logits, loss, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


class FullModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_in1 = nn.Linear(config.input_size_x, config.n_embd)
        self.lm_in2 = nn.Linear(config.input_size_z, config.n_embd)
        self.wpe = nn.Linear(2 * config.n_embd, config.n_embd)
        self.block = Block(config)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.output_size, bias=False)
        self.apply(_init_weights)

    def forward(self, x, z, targets):
        x = self.lm_in1(x)
        z = self.lm_in2(z)
        xz = torch.cat((x, z), 1)
        xz = self.wpe(xz)
        xz = self.block(xz)
        logits = self.lm_head(xz)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets, ignore_index=-1)

        return logits, loss, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


class TramModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_in1 = nn.Linear(config.input_size_x, config.n_embd)
        self.lm_in2 = nn.Linear(config.input_size_z, config.n_embd)
        self.wpe1 = nn.Linear(config.n_embd, config.n_embd)
        self.block1 = Block(config)
        self.wpe2 = nn.Linear(2 * config.n_embd, config.n_embd)
        self.block2 = Block(config)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        # no PI head
        self.lm_head1 = nn.Linear(
            config.n_embd,
            config.output_size,
            bias=False
        )
        # PI head
        self.lm_head2 = nn.Linear(
            config.n_embd,
            config.output_size,
            bias=False
        )
        self.apply(_init_weights)

    def forward(self, x, z, targets):
        x = self.lm_in1(x)
        z = self.lm_in2(z)

        x = self.wpe1(x)
        x = self.block1(x)
        logits1 = self.lm_head1(x.detach())  # stop gradient from NO PI head
        loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), targets, ignore_index=-1)

        xz = torch.cat((x, z), dim=1)
        xz = self.wpe2(xz)
        xz = self.block2(xz)
        logits2 = self.lm_head2(xz)
        loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), targets, ignore_index=-1)

        return logits1, loss1 + loss2, loss1

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
