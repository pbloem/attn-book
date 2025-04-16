import fire

import torch
from torch import nn
import torch.nn.functional as F
import math, random

import attnbook
from attnbook.data import load_imdb

from tqdm import tqdm

import optuna

BACKENDS = torch.cuda, torch.backends.mps

"""
Creates the basic models discussed in the first two chapters of the book:
- An embedding model 
- An embedding model enriched with a simple self-attention layer
- An embedding model enriched with a multi-head selt-attention layer.
"""

def attention(queries, keys, values):

    assert keys.size() == values.size()
    assert queries.size()[-1] == keys.size()[-1]   # embedding dimension
    assert queries.size()[:-2] == keys.size()[:-2] # batch dimensions

    k = queries.size(-1)

    raw_weights = queries @ keys.transpose(-2, -1)

    raw_weights = raw_weights / math.sqrt(k)
    weights = raw_weights.softmax(dim=-1)

    # print(weights.size(), values.size()); exit()

    return weights @ values
    # -- In the book I have it the other way around. Is that better somehow?

class SimpleSelfAttention(nn.Module):

    def forward(self, x):
        return attention(x, x, x)

class MHSelfAttention(nn.Module):

    def __init__(self, k, heads):
        super().__init__()

        assert k % heads == 0

        self.heads = heads
        self.tokqv = nn.Linear(k, 3*k)

        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):

        b, t, k = x.size()
        h = self.heads
        s = k // h

        kqv = self.tokqv(x)
        kqv = kqv.view(b, t, h, 3 * s)
        kqv = kqv.transpose(1,2)

        res = attention(*kqv.split(s, dim=-1))

        res = res.transpose(1,2).reshape(b, t, k)

        return self.unifyheads(res)

class EmbeddingModel(nn.Module):
    """
    A basic embedding model. Word embeddings, followed by a mean pooling an a classification layer.

    Optionallly includes a mixer (self-attention ) and a feedforward layer.
    """

    def __init__(self, v, k, cls, mixer=None, ff=False):
        super().__init__()

        self.embed = nn.Embedding(v, k)
        self.tocls = nn.Linear(k, cls)

        self.mixer = mixer

        self.ff = nn.Sequential(
            nn.Linear(k, 4*k), nn.ReLU(),
            nn.Linear(4*k, k)
        ) if ff else None


    def forward(self, x, mask=None):

        x = self.embed(x)

        x = x if self.mixer is None else x + self.mixer(x) # self-attention

        x = self.ff(x) if self.ff is not None else x

        if mask is not None:
            x = x * mask
        x = x.mean(dim=1)

        return self.tocls(x)

def make_batches(x_data, y_data, pad_token, batch_tokens):
    """
    Returns a list of batches from the given data.

    :param x_data:
    :param y_data:
    :param batch_tokens:
    :param i2w:
    :return: A list of triples (x, mask, y) with each elements a tensor.
    """

    # Sort in reverse length order
    x_data, y_data = zip(*sorted(zip(x_data, y_data),
                              key = lambda x : -len(x[0])))

    # Batching code
    batches = []
    current = 0
    while current < len(x_data):
        max_len = len(x_data[current])
        batch_size = max(batch_tokens // max_len, 1)

        x_batch, y_batch = x_data[current : current + batch_size], y_data[current : current + batch_size]

        mask    = [ [1.] * len(x) + [0.] * (max_len - len(x))           for x in x_batch]
        x_batch = [ x             + [pad_token] * (max_len - len(x))    for x in x_batch]

        x_batch, mask, y_batch = torch.tensor(x_batch), torch.tensor(mask), torch.tensor(y_batch)

        batches.append((x_batch, mask, y_batch))

        current = current + batch_size

    assert sum(b[0].size(0) for b in batches) == len(x_data)
    return batches

def go(emb=300, epochs=3, batch_tokens=10_000, lr=3e-4, mixer=None, ff=False, heads=4):

    print('Loading data. ', end='')
    (x_train, y_train), (x_val, y_val), (i2w, w2i), cls = load_imdb(final=False, char=False)
    print('Done.')
    v = len(i2w) # Vocabulary size

    train = make_batches(x_train, y_train, w2i['.pad'], batch_tokens)
    valid = make_batches(x_val, y_val, w2i['.pad'], batch_tokens)

    # Create model
    if mixer == 'simple':
        mx = SimpleSelfAttention()
    elif mixer == 'mh':
        mx = MHSelfAttention(emb, heads=heads)
    else:
        mx = None

    model = EmbeddingModel(v, emb, cls=cls, mixer=mx, ff=ff)
    if torch.cuda.is_available(): model.cuda()

    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    # Training loop
    for e in range(epochs):

        random.shuffle(train)

        for x, m, y in (bar := tqdm(train)):

            opt.zero_grad()
            if torch.cuda.is_available():
                x, m, y = x.to('cuda'), m.to('cuda'), y.to('cuda')

            out = model(x)
            loss = F.cross_entropy(out, y)

            loss.backward()
            opt.step()

            bar.set_postfix({'loss': loss.item()})

    # Eval
    correct = num = 0.
    for x, m, y in (bar := tqdm(valid)):

        if torch.cuda.is_available():
            x, m, y = x.cuda(), m.cuda(), y.cuda()

        out = model(x)

        correct += (out.argmax(dim=-1) == y).sum().item()
        num += x.size(0)

    return {'accuracy' : correct/num}

def tune_go(trial : optuna.Trial, mixer):

    res = go(
        epochs = 6,
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        emb = trial.suggest_categorical('emb', [16, 32, 64, 128, 256, 512]),
        ff = trial.suggest_categorical('ff', [True, False]),
        batch_tokens = 50_000,
        heads = 4,
        mixer = mixer
    )

    return res['accuracy']

def tune(mixers=['none', 'simple', 'mh'], trials=100):

    for mixer in mixers:

        print('Trial with mixer:', mixer)
        study = optuna.create_study(
            storage=f'sqlite:///db.sqlite3',  # Specify the storage URL here.
            study_name=f'tune-{mixer}',
            load_if_exists=True,
            direction="maximize",
        )

        study.optimize(lambda x : tune_go(x, mixer), n_trials=trials )

        print(f'Finished ({mixer}). Result:')
        print('\t', study.best_params)

if __name__ == '__main__':
    fire.Fire()

