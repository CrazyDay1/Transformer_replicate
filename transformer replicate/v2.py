import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters 
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for prediction (input)
max_iters = 5000
eval_interval = 300 # get mean loss for whole model to reduce noise 
learning_rate = 1e-3 # learning rate is higher due to small scale of model 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 # number of embedding dimentions 
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all unique characters in the dataset 
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integer (encode/ decode)
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # take a string and output a list of integer 
decode = lambda l: ''.join([itos[i] for i in l]) # take a list of integer and outputs a string 

# Train test split 
data = torch.tensor(encode(text), dtype = torch.long)
n = int(.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)- block_size, (batch_size,)) # sample random sequence(blocks) from input 
    # stack the sequence to form a batch 
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y

# Everything in the function below will not initiate a backward propagation, more memory efficient because variables are not stored. 
@torch.no_grad()
# this function is not optimised yet, behave the same in training and inference, could be improved with drop out, batch norm etc
def estimate_loss():
    out = {}
    model.eval() # has no function yet since model behave same in training and inference 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): # function is introduced to get average of loss in the model, reducing noise 
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # has no function yet since model behave same in training and inference 
    return out 

class Head(nn.Module): 
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)# (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B,T,T)
        wei = F.softmax(wei, dim = -1) # (B,T,T)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out 
    
class MultipleHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim = -1)  
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
 
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a look up table 
        self.token_embedding_table =  nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.sa_head = MultipleHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self attention  
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) 

    def forward(self, idx, targets = None):
        B,T = idx.shape 

        # idx and targets are both (B, T) tensors of integers 
        tok_emb = self.token_embedding_table(idx) # (B, T, C) 
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.sa_head(x) # apply one head of self-attention. (B,T,C)
        x = self.ffwd(x) # (B,T,C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets == None:
            loss = None
        
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss 
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indicies in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens 
            idx_cond = idx[:, -block_size:]
            # get the predictions 
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1)
            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples= 1) # (B, 1)
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model 
context = torch.zeros((1,1), dtype= torch.long, device= device)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))