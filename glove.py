from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

from dataset import Dataset


class Glove(nn.Module):
    def __init__(self, vocab_size, embed_size, x_max=5, alpha=0.75):
        super(Glove, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.x_max = x_max
        self.alpha = alpha

        self.l_vecs = nn.Embedding(self.vocab_size, self.embed_size)
        self.l_vecs.weight.data.normal_(std=0.01)
        self.r_vecs = nn.Embedding(self.vocab_size, self.embed_size)
        self.r_vecs.weight.data.normal_(std=0.01)

        self.l_bias = nn.Embedding(self.vocab_size, 1)
        self.l_bias.weight.data.normal_(std=0.01)
        self.r_bias = nn.Embedding(self.vocab_size, 1)
        self.r_bias.weight.data.normal_(std=0.01)

    def forward(self, x):
        raise NotImplementedError()

    def fn(self, x):
        if x < self.x_max:
            return (x / self.x_max)**self.alpha
        return 1

    def loss(self, i, j, count):
        l = self.l_vecs(i)
        r = self.r_vecs(j)
        l_bias = self.l_bias(i)
        r_bias = self.r_bias(j)

        # Element-wise dot product followed by bias and log terms
        out = (torch.mm(l, r.t()).diag() + l_bias + r_bias - torch.log(count))**2
        out = torch.mul(out, self.fn(count))

        return out.sum()


def train_glove(args):
    # Get the data loader
    dataset = Dataset(args.input, args.context_size, args.vocab_min_count)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    # Create a Glove model instance
    model = Glove(dataset.vocab_size, args.embed_size)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training
    for epoch in range(args.num_epochs):
        average_loss = 0

        for i, (l, r, count) in enumerate(data_loader):
            l, r, count = Variable(l), Variable(r), Variable(count.float())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            loss = model.loss(l, r, count) / l.size(0)
            loss.backward()
            optimizer.step()

            average_loss += loss.data[0]

        print('Average loss after Epoch [%d/%d]: %.4f' % (epoch + 1, args.num_epochs, average_loss / len(data_loader)))

    # Save the word vectors to a file
    print('Writing vectors to {}...'.format(args.output))
    f = open(args.output, 'w')
    for i, w in enumerate(dataset.vocab):
        s = w

        j = Variable(torch.LongTensor([i]))
        vec = (model.l_vecs(j) + model.r_vecs(j)).data.squeeze()
        for k in range(vec.size(0)):
            s += ' ' + str(round(vec[k], 6))

        f.write(s)

    f.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', default='wiki_data.txt', help='Input text file')
    parser.add_argument('--output', default='vectors.txt', help='Output file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs through the data')
    parser.add_argument('--context_size', type=int, default=3, help='Window size')
    parser.add_argument('--embed_size', type=int, default=50, help='Word vector size')
    parser.add_argument('--vocab_min_count', type=int, default=5, help='Min number of word occurances required')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    train_glove(args)
