import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder

    def loss(self, sample):
        xs = Variable(sample["xs"])  # support
        xq = Variable(sample["xq"])  # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = (
            torch.arange(0, n_class)
            .view(n_class, 1, 1)
            .expand(n_class, n_query, 1)
            .long()
        )
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat(
            [
                xs.view(n_class * n_support, *xs.size()[2:]),
                xq.view(n_class * n_query, *xq.size()[2:]),
            ],
            0,
        )

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[: n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support :]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {"loss": loss_val.item(), "acc": acc_val.item()}


@register_model("protonet_conv")
def load_protonet_conv(**kwargs):
    x_dim = kwargs["x_dim"]
    hid_dim = kwargs["hid_dim"]
    z_dim = kwargs["z_dim"]

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten(),
    )

    return Protonet(encoder)

@register_model("protonet_text")
def load_protonet_text(**kwargs):
    x_dim = kwargs["x_dim"]
    hid_dim = kwargs["hid_dim"]
    z_dim = kwargs["z_dim"]

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    encoder = nn.Sequential(
        nn.Embedding(128003, 28),
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten(),
    )

    return Protonet(encoder)

@register_model("protonet_lstm")
def load_protonet_lstm(**kwargs):
    vocab_file = kwargs["vocab_file"]
    embed_size = kwargs["embed_size"]
    output_size = 64

    def read_vocab_from_file(vocab_file):
        vocab = {}
        with open(vocab_file) as fd:
            for id, token in enumerate(fd):
                vocab[token.strip()] = id

        return vocab

    def token_stream_to_id(vocab_dict, token_stream_list):
        token_id_list = []
        for token_stream in token_stream_list:
            token_id_list.append([vocab_dict[token] for token in token_stream])

        return token_id_list

    def token_id_to_tensor(x):
        return torch.as_tensor(np.asarray(x, dtype=np.int))

    vocab = read_vocab_from_file(vocab_file)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            token_ids = token_stream_to_id(vocab, x)
            x = token_id_to_tensor(token_ids)

            lstm = nn.LSTM(embed_size, output_size)
            embed_layer = nn.Embedding(len(vocab), embed_size)

            x = embed_layer(x)
            out, (hidden, cell) = lstm(x)

            return out[:, -1, :]

    encoder = Model()

    return Protonet(encoder)
