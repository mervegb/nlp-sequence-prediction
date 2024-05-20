import torch
import torch.nn as nn
from torch.nn import functional as F


# Batch Size: number of training examples used to train single iteration of model
# Block Size: sequence length determines how many previous words or tokens the model considered before making prediction

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
max_iters = 3000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32

# eval_iters: number of iterations used to evaluate the model's performance
# this is used during training loop to periodically check how well the model is performing on data it hasn't seen before
# this helps to prevent overfitting to ensure the model generalizes well to new data


torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]


# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train and rest will be validation
train_data = data[:n]
val_data = data[n:]


# Data Loading
def get_batch(split):
    # generate small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# it reduces memory usage and speeds up computations since gradients are not needed
@torch.no_grad()
def estimate_loss():
    out = {}  # output dictionary, it will store average loss for both training & validation splits
    model.eval()  # sets the model to evaluation mode, this disables certain layers like batch normalization and dropout that behave differently during training and evaluation
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()  # resets the model back to training mode
    return out


# Bigram Model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # initialize embedding layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

       # linear layer that transforms embeddings back to the vocabulary size
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        logits = self.lm_head(tok_emb)  # (B,T, vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            # reshapes logits from (B,T,C) to (B*T,C) to flatten the batch and sequence dimensions into one
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # this method generates new tokens
    # idx tensor of shape (B,T) representing the current context, B batch size, T sequence length
    # max_new_tokens number of new tokens to generate
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # calls the forward pass to get predictions and loss
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # becomes (B,C)
            probs = F.softmax(logits, dim=-1)

            # (B,1) this contains the indices of the newly generated tokens
            idx_next = torch.multinomial(probs, num_samples=1)
            # extends the current sequences by adding the newly generated tokens (B,T+1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create Pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


# Training loop
for iter in range(max_iters):
    # every once in a while we evaluate the loss on train and validation sets
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    # clears old gradients from previous steps
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # performs backpropagation
    optimizer.step()  # updates model's parameters using the computed gradients


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
