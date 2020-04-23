from protonets.models import few_shot


def test_lstm(datadir):
    kwargs = {"vocab_file": datadir / "vocab.txt", "embed_size": 300}
    encoder = few_shot.load_protonet_lstm(**kwargs).encoder

    x = ["习近平同美国总统特朗普通电话", "习近平同美国总统特朗普通电话"]

    ret = encoder.forward(x)
    import pdb

    pdb.set_trace()
