import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# 1 使用 tiktoken 分词
# ---------------------------
ENC = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = ENC.n_vocab

class Tok:
    def __init__(self, enc):
        self.enc = enc
        self.vocab_size = enc.n_vocab
    def encode(self, s):
        return self.enc.encode(s)
    def decode(self, ids):
        return self.enc.decode(list(map(int, ids)))

tokenizer = Tok(ENC)

# ---------------------------
# 2 数据集：将 token 列表按比例切分为 train/val/test
#    然后每段用滑动窗口
# ---------------------------
class IdsWindowDataset(Dataset):
    """
    ids: token id 列表
    context_len: 输入序列长度
    stride: 滑窗步长
    """
    def __init__(self, ids, context_len, stride = 1):
        super().__init__()
        self.ids = ids
        self.context_len = context_len
        self.stride = max(1, int(stride))
        # 最后一个起始位置使得可以取到 context_len + 1 长度（要 x 和 y）
        max_start = max(0, len(ids) - (context_len + 1))
        self.starts = list(range(0, max_start + 1, self.stride))
        if len(self.starts) == 0:
            raise ValueError("数据划分（train/val/test）太短，无法构造样本。请使用更长文本或减少 context_len。")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        chunk = self.ids[s : s + self.context_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ---------------------------
# 3 模型（embedding + pos emb + transformer block）
# ---------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias

class MLP(nn.Module):
    def __init__(self, dim, mult=4, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(drop)
        )
    def forward(self, x):
        return self.net(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, context_len, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape -> [B, heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, T, T]
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # [B, heads, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, context_len, mlp_mult=4, drop=0.1):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, context_len, attn_drop=drop, proj_drop=drop)
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim, mult=mlp_mult, drop=drop)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, context_len=128, dim=256, n_layers=4, n_heads=8, drop=0.1):
        super().__init__()
        self.context_len = context_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(context_len, dim)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, context_len, mlp_mult=4, drop=drop) for _ in range(n_layers)])
        self.ln_f = LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):  # idx: [B, T]
        B, T = idx.shape
        assert T <= self.context_len
        tok = self.tok_emb(idx)  # [B, T, C]
        pos = self.pos_emb(torch.arange(T, device=idx.device))[None, :, :]
        x = self.drop(tok + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]
        return logits

# ---------------------------
# 4 训练 验证 测试 生成 
# ---------------------------
def plot_and_save(train_x, train_y, val_x, val_y, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(train_x, train_y, label="train")
    if val_x and val_y:
        plt.plot(val_x, val_y, label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None, device='cpu'):
    prev_mode = model.training
    model.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.context_len:]
            logits = model(idx_cond)[:, -1, :]
            logits = logits / (temperature + 1e-12)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k=k)
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
    generated_ids = idx[0].tolist()[len(ids):]
    out = tokenizer.decode(generated_ids)
    if prev_mode:
        model.train()
    else:
        model.eval()
    return out

# ---------------------------
# 5 主函数：参数解析 数据分割 训练流程
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--context_len', type=int, default=128)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--log_path', type=str, default='loss_curve.png')
    parser.add_argument('--sample', type=str, default=None, help='训练结束后直接生成样例文本（提供 prompt）')
    parser.add_argument('--sample_len', type=int, default=120)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--generate', action='store_true', help='仅生成（不训练），需要 --ckpt 提供 checkpoint')
    parser.add_argument('--ckpt', type=str, default=None, help='加载指定 checkpoint 用于继续训练或生成')
    parser.add_argument('--prompt', type=str, default='Hello', help='生成时的起始 prompt（仅在 --generate 或 --sample 时使用）')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("device =", device)
    print("vocab size =", VOCAB_SIZE)

    # 仅生成模式：加载模型并生成
    if args.generate:
        assert args.ckpt is not None and os.path.exists(args.ckpt), "--generate 需要存在的 --ckpt"
        model = SimpleGPT(VOCAB_SIZE, context_len=args.context_len, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads)
        model.to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        print("已加载 checkpoint:", args.ckpt)
        out = generate_text(model, tokenizer, args.prompt, max_new_tokens=args.sample_len, temperature=args.temperature, top_k=args.top_k, device=device)
        print("\n==== Generated Text ====\n")
        print(out)
        return

    # 训练模式：需要 --train_file
    assert args.train_file is not None and os.path.exists(args.train_file), "训练模式需要存在的 --train_file"
    with open(args.train_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 整体编码为 token ids
    all_ids = tokenizer.encode(text)
    N = len(all_ids)
    assert N > args.context_len + 1, "文本太短，无法构造样本；请提供更长的训练文本或减小 context_len"

    # 按序列切分 train/val/test
    train_end = int(N * args.train_frac)
    val_end = train_end + int(N * args.val_frac)
    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids = all_ids[val_end:]

    print(f"tokens total={N}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # 构造 dataset 和 dataloader
    train_ds = IdsWindowDataset(train_ids, context_len=args.context_len, stride=args.stride)
    val_ds = IdsWindowDataset(val_ids, context_len=args.context_len, stride=args.stride) if len(val_ids) > args.context_len + 1 else None
    test_ds = IdsWindowDataset(test_ids, context_len=args.context_len, stride=args.stride) if len(test_ids) > args.context_len + 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False) if test_ds is not None else None

    # 构建模型与优化器
    model = SimpleGPT(VOCAB_SIZE, context_len=args.context_len, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 尝试加载 checkpoint（可选继续训练）
    start_epoch = 1
    if args.ckpt and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck['model'])
        if 'epoch' in ck:
            start_epoch = ck['epoch'] + 1
        print("Loaded checkpoint", args.ckpt, "start_epoch", start_epoch)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    # 训练日志用于绘图
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []

    global_step = 0
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        running_steps = 0
        for xb, yb in train_loader:
            global_step += 1
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)  # [B, T, V]
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            lval = loss.item()
            running_loss += lval
            running_steps += 1

            # 每 10 step 记录一次训练 loss 并保存图像
            if global_step % 10 == 0:
                train_steps.append(global_step)
                train_losses.append(lval)
                plot_and_save(train_steps, train_losses, val_steps, val_losses, args.log_path)

        avg_epoch_loss = running_loss / max(1, running_steps)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} done. avg_loss={avg_epoch_loss:.4f} time={(t1-t0):.1f}s")

        # 保存 checkpoint
        torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(args.ckpt_dir, f'epoch_{epoch}.pt'))
        torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(args.ckpt_dir, 'last.pt'))

        # 验证集评估
        if val_loader is not None:
            model.eval()
            v_loss_sum = 0.0
            v_steps = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    logits = model(xb)
                    vloss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                    v_loss_sum += vloss.item(); v_steps += 1
            avg_val_loss = v_loss_sum / max(1, v_steps)
            val_steps.append(global_step)
            val_losses.append(avg_val_loss)
            plot_and_save(train_steps, train_losses, val_steps, val_losses, args.log_path)
            print(f"  Validation loss: {avg_val_loss:.4f}")
            model.train()

    print("Training finished. Checkpoints saved to", args.ckpt_dir)
    print("Loss curve saved to", args.log_path)

    # 训练结束后在 test 集上评估
    if test_loader is not None:
        model.eval()
        tst_sum = 0.0; tst_steps = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                tloss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                tst_sum += tloss.item(); tst_steps += 1
        print("Test loss:", (tst_sum / max(1, tst_steps)))

    # 生成展示
    if args.sample is not None:
        print("Generating sample with prompt:", args.sample)
        out = generate_text(model, tokenizer, args.sample, max_new_tokens=args.sample_len, temperature=args.temperature, top_k=args.top_k, device=device)
        print("\n==== Generated Text ====\n")
        print(out)

if __name__ == "__main__":
    main()