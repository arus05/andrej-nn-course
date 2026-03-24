import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { c:i for i, c in enumerate(chars)}
itos = { i:c for i, c in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join([itos[id] for id in ids])

data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train = data[:split]
valid = data[split:]

torch.manual_seed(100)

# Hyperparameters
block_size = 256
batch_size = 64
dropout = 0.2
n_embd = 384
eval_iters = 200
max_iters = 5000
eval_interval = 500
n_blocks = 6
n_heads = 6
head_size = 32
learning_rate = 3e-4
# -----

block_size = 4
batch_size = 8
dropout = 0.2
n_embd = 32
eval_iters = 100
max_iters = 500
eval_interval = 50
n_blocks = 4
n_heads = 4
head_size = 8
learning_rate = 1e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(split):
  assert split in ['train', 'valid'], "split is either train or valid"

  if split == "train":
    data = train
  else:
    data = valid

  idxs = torch.randint(0, len(data)-block_size, (batch_size, ))

  x = torch.stack([data[idx:idx+block_size] for idx in idxs])
  y = torch.stack([data[idx+1:idx+block_size+1] for idx in idxs])

  x, y = x.to(device), y.to(device)

  return x, y
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, n_embd * 4),
      nn.ReLU(),
      nn.Linear(n_embd * 4, n_embd),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.q = nn.Linear(n_embd, head_size, bias=False)
    self.k = nn.Linear(n_embd, head_size, bias=False)
    self.v = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('mask', torch.tril(torch.ones((block_size, block_size))))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape

    Q = self.q(x) # (B, T, head_size)
    K = self.k(x) # (B, T, head_size)
    V = self.v(x) # (B, T, head_size)

    weights = (Q @ K.transpose(-2, -1)) * Q.shape[-1]**-0.5 # (B, T, T)
    weights = weights.masked_fill(self.mask[:T, :T]==0, float('-inf'))
    weights = F.softmax(weights, dim=-1)
    weights = self.dropout(weights)
    att_vals = weights @ V # (B, T, head_size)

    return att_vals

class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.proj = nn.Linear(n_heads * head_size, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = torch.cat([head(x) for head in self.heads], dim=-1) # (B, T, n_heads * head_size)
    x = self.proj(x) # (B, T, n_embd)
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self, n_embd, n_heads):
    super().__init__()
    self.sa = MultiHeadAttention(n_heads, n_embd//n_heads)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    # x -> (B, T, n_embd)
    x = x + self.sa(x + self.ln1(x)) # (B, T, n_embd)
    x = x + self.ffwd(self.ln2(x)) # (B, T, n_embd)
    return x

class LayerNorm():
  def __init__(self, dim, eps = 1e-5):
    # dim = n
    self.eps = eps
    self.gamma = torch.ones(dim) # (dim)
    self.beta = torch.zeros(dim) # (dim)

  def __call__(self, x):
    # x -> (n_rows, n_cols)
    mean = x.mean(dim=1, keepdim=True) # (n_rows, 1)
    var = x.var(dim=1, keepdim=True) # (n_rows, 1)
    x = (x - mean) / (var + self.eps)**0.5
    x = x * self.gamma + self.beta # ()
    return x


class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, n_embd) # (vocab_size, n_embd)
    self.positional = nn.Embedding(block_size, n_embd) # (block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_blocks)])
    self.ln = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, x, targets = None):
    # x -> (B, T), targets -> (B, T)
    B, T = x.shape
    token_embd = self.embedding(x) # (B, T, n_embd)
    pos_embd = self.positional(torch.arange(T, device=device)) # (T, n_embd)
    x = token_embd + pos_embd # (B, T, n_embd)
    x = self.blocks(x) # (B, T, n_embd)
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      logits = logits.view(B*T, vocab_size) # (B*T, vocab_size)
      targets = targets.view(B*T) # (B*T, )
      loss = F.cross_entropy(logits, targets) # scalar

    return logits, loss

  def generate(self, x, max_new_tokens=10):
    # x -> (B, T)
    for _ in range(max_new_tokens):
      cropped_x = x[:, -block_size:]
      logits, _ = self(cropped_x) # (B, T, vocab_size)
      """
        for each sample in the batch, look at the last character's logits and choose the next character.
        append the character at the end of each sample
      """
      logits = logits[:, -1, :] # (B, vocab_size)
      probs = F.softmax(logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
      x = torch.cat((x, next_token), dim=-1) # (B, T+1)

    return x

m = BigramLanguageModel().to(device)

@torch.no_grad()
def compute_loss():
  # prediction -> (B, T, vocab_size)
  # actual -> (B, T)
  result = {}
  m.eval()

  for split in ['train', 'valid']:
    losses = torch.zeros((eval_iters,))
    for i in range(eval_iters):
      xb, yb = get_batch(split)
      _, loss = m(xb, yb)
      losses[i] = loss.item()
    result[split] = losses.mean()

  m.train()
  return result

# training
optimizer = optim.NAdam(m.parameters(), lr=learning_rate)

for i in range(max_iters):
  x, y = get_batch("train")

  # reset gradients
  optimizer.zero_grad(set_to_none=True)

  # forward pass
  logits, loss = m(x, y)

  # backward pass
  loss.backward()

  # update
  optimizer.step()

  # log statistics
  # print(f"Step {i+1} | loss = {loss}")
  if i % eval_interval == 0:
    losses = compute_loss()
    print(f"Step {i}:  Train loss: {losses['train']},  Val loss: {losses['valid']}")

# generate output
def generate():
    xt = torch.tensor([encode("a")]).to(device)
    yt = m.generate(xt, 1000)

    outputs = [decode(output.tolist()) for output in yt]
    for output in outputs:
        print(output)