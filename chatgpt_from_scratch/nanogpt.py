import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


#hyperparameters
torch.manual_seed(1337)
block_size = 256
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

#load dataset
with open('tinyshakespeare.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
print('length of dataset in chars:', len(text))

#unique chars in text
chars = sorted(list(set(text)))
vocab_size = (len(chars))

#mapping from chars <-> integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #returns ints from strings
decode = lambda l: ''.join([itos[c] for c in l]) #returns strings from ints

data = torch.tensor(encode(text), dtype=torch.long)

#set up train, validate data splits
n = int(len(data)*0.9)

train_data = data[:n]
val_data = data[n:]



def get_batch(split):
    data = train_data if split=='train' else val_data
    #get the ix randomly
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) 
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape #batch, time, channels

        #single attention head
        k = self.key(x)    # (B,T,C) @ (C, head_size) ---> (B,T, head_size) #key results
        q = self.query(x)  # (B,T,C) @ (C, head_size) ---> (B,T, head_size) #value results

        #dot product the queries with the keys
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B,T,head_size) @ (B,head_size,T) ----> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) #  (B,T, head_size)
        out = wei @ v # (B,T,T) @ (B,T, head_size)  ---> (B,T,head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1) #concatenating over the channel dimension
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """transformer block: intersperses communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb #(B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        if targets == None:
            loss = None
        else:
            B, T, C= logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets) #cross entropy expects (B, C) shape
        
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        #initial context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens so positional embeddings don't blow up
            idx_cond = idx[:, -block_size:] 
        
            logits, loss = self(idx_cond) # get predictions
            logits = logits[:, -1, :] #focus only on the last timestep 

            probs = F.softmax(logits, dim=-1) #use softmax to get probabilities

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) one prediction for each batch item

            idx = torch.cat((idx, idx_next), dim=1) # (B, t+1)
        return idx
    

    
model = BigramLanguageModel()
m=model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):

    #every once in a while evaluate our loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sample data batch
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=300)[0].tolist()))