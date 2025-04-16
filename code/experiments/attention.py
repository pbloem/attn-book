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
MAX_LENGTH = 3000
"""
Creates the basic models discussed in the first two chapters of the book:
- An embedding model 
- An embedding model enriched with a simple self-attention layer
- An embedding model enriched with a multi-head selt-attention layer.
"""


def mask_batch(inputs=None,
               num_tokens=32768,
               mlm_probability=.15,
               use_80_20_rule=True,
               mask_token=0,
            ):
        labels = inputs.clone() # prediction target, the unmasked input
        # -- NB non-manipulated tokens are masked out below (by setting the target to -100).

        number_of_masks = round(mlm_probability * inputs.shape[1])
        mask_locations = torch.argsort(torch.randint_like(inputs, inputs.shape[1]))[:, :number_of_masks]
        # -- this was slightly fudged to be faster. A draw of torch.rand would be more random, but take slightly longer to sort

        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        masked_indices.scatter_(1, mask_locations, 1)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # -- Note that -100 is the default ignore_index in the CrossEntropyLoss
        #    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        if use_80_20_rule:
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            first_80percent_mask_locations = mask_locations[:, : round(0.8 * number_of_masks)]

            indices_replaced = torch.zeros_like(inputs, dtype=torch.bool)
            indices_replaced.scatter_(1, first_80percent_mask_locations, 1)
            inputs[indices_replaced] = mask_token

            # 10% of the time, we replace masked input tokens with random word
            next_10percent_mask_locations = mask_locations[:, round(0.8 * number_of_masks) : round(0.9 * number_of_masks)]

            indices_random = torch.zeros_like(inputs, dtype=torch.bool)
            indices_random.scatter_(1, next_10percent_mask_locations, 1)

            random_words = torch.randint(num_tokens, labels.shape, dtype=inputs.dtype, device=inputs.device)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            # -- Note that these are different from the unmasked tokens in that we _do_ compute a loss over them.
            pass
        else:
            # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            inputs[masked_indices] = mask_token

        return inputs, labels

def attention(queries, keys, values):

    assert keys.size() == values.size()
    assert queries.size()[-1] == keys.size()[-1]   # embedding dimension
    assert queries.size()[:-2] == keys.size()[:-2] # batch dimensions

    k = queries.size(-1)

    raw_weights = queries @ keys.transpose(-2, -1)

    raw_weights = raw_weights / math.sqrt(k)
    weights = raw_weights.softmax(dim=-1)

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

        res = res.transpose(1,2)
        res = res.reshape(b, t, k)

        return self.unifyheads(res)

class EmbeddingModel(nn.Module):
    """
    A basic embedding model. Word embeddings, followed by a mean pooling an a classification layer.

    Optionallly includes a mixer (self-attention ) and a feedforward layer.
    """

    def __init__(self, v, k, cls, mixer='simple', ff=False, layers=0, heads=4, pos=False, aux=False):
        super().__init__()

        self.embed = nn.Embedding(v, k)
        self.pos = nn.Embedding(embedding_dim=k, num_embeddings=MAX_LENGTH) if pos else None
        self.tocls = nn.Linear(k, cls)

        ls = []

        for _ in range(layers):
            ls.append(make_mixer(mixer, k, heads))

            if ff:
                ls.append(nn.Sequential(
                    nn.Linear(k, 4*k), nn.ReLU(),
                    nn.Linear(4*k, k)
                ))

        self.layers = nn.Sequential(*ls)

        self.aux = nn.Linear(k, v) if aux else None

    def forward(self, x, mask=None):

        x = self.embed(x)

        b, t, k = x.size()

        if self.pos is not None:
            pos = self.pos(torch.arange(t, device=x.device))[None, :, :].expand(b, t, k)
            x = x + pos

        x = self.layers(x)

        if mask is not None:
            p = x * mask
        else:
            p = x
        p = p.mean(dim=1) # pooled

        cls = self.tocls(p)

        if self.aux is None:
            return cls, None
        return cls, self.aux(x)

def make_mixer(mixer, emb=None, heads=None):
    # Create model
    if mixer == 'simple':
        return SimpleSelfAttention()

    if mixer == 'mh':
        return MHSelfAttention(emb, heads=heads)

    else:
        raise

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

def go(emb=300,
       vocab=20_000,
       epochs=3,
       batch_tokens=3_000,
       lr=3e-4, mixer='simple',
       layers=0,
       ff=False,
       heads=4,
       pos=False,
       aux=False # Whether to include an auxiliary loss term (i.e. masking)
       ):

    print('Loading data. ', end='')
    (x_train, y_train), (x_val, y_val), (i2w, w2i), cls = load_imdb(voc=vocab, final=False, char=False)
    print('Done.')
    v = len(i2w) # Vocabulary size

    train = make_batches(x_train, y_train, w2i['.pad'], batch_tokens)
    valid = make_batches(x_val, y_val, w2i['.pad'], batch_tokens)

    model = EmbeddingModel(v, emb, cls=cls, mixer=mixer,
                           ff=ff, layers=layers, heads=heads,
                           pos=pos, aux=aux)
    print(model)

    if torch.cuda.is_available(): model.cuda()

    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    # Training loop
    for e in range(epochs):

        random.shuffle(train)

        for x, m, y in (bar := tqdm(train)):

            opt.zero_grad()
            if torch.cuda.is_available():
                x, m, y = x.to('cuda'), m.to('cuda'), y.to('cuda')

            if aux:
                with torch.no_grad():
                    x, my = mask_batch(x, num_tokens=v, mask_token=w2i['.unk'])

            cout, mout = model(x)

            closs = F.cross_entropy(cout, y)
            if aux:
                mloss = F.cross_entropy(mout.transpose(1, 2), my)
                loss = closs + mloss
            else:
                loss = closs

            loss.backward()
            opt.step()

            bar.set_postfix({'closs': closs.item()})

    # Eval
    correct = num = 0.
    for x, m, y in (bar := tqdm(valid)):

        if torch.cuda.is_available():
            x, m, y = x.cuda(), m.cuda(), y.cuda()

        out, _ = model(x)

        correct += (out.argmax(dim=-1) == y).sum().item()
        num += x.size(0)

    return {'accuracy' : correct/num}

def tune_go(trial : optuna.Trial):

    res = go(
        epochs = 6,
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        emb = trial.suggest_categorical('emb', [16, 32, 64, 128, 256, 512]),
        ff = trial.suggest_categorical('ff', [True, False]),
        pos = trial.suggest_categorical('pos', [True, False]),
        aux = trial.suggest_categorical('aux', [True, False]),
        batch_tokens = 50_000,
        heads = 4,
        mixer = trial.suggest_categorical('mixer', ['simple', 'mh']),
        layers = trial.suggest_categorical('layers', [0,1,2,3]),
    )

    return res['accuracy']

def tune(trials=100, name='tune-attention'):

    study = optuna.create_study(
        storage=f'sqlite:///db.sqlite3',  # Specify the storage URL here.
        study_name=name,
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(tune_go, n_trials=trials )

    print(f'Finished. Result:')
    print('\t', study.best_params)

if __name__ == '__main__':
    fire.Fire()

