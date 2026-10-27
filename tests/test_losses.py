import torch

from hemahier.losses import gather_on_chain, ramp, ranking_loss


def test_ranking_zero_when_ordered_with_margin():
    s = torch.tensor([0.0, 0.3, 0.6, 0.1, 0.9])
    pi = torch.tensor([0.0, 0.5, 1.0, 0.0, 1.0])
    q = torch.tensor([0, 0, 0, 1, 1])
    assert ranking_loss(s, pi, q, margin=0.1) == 0


def test_ranking_penalises_inversion_only_within_chain():
    s = torch.tensor([0.8, 0.2, 0.0])
    pi = torch.tensor([0.0, 1.0, 0.5])
    q = torch.tensor([0, 0, 1])          # sample 2 is on another chain: no pair with it
    assert torch.isclose(ranking_loss(s, pi, q, margin=0.1, reduction="sum"), torch.tensor(0.7))


def test_gather_on_chain_skips_off_chain(ont):
    y = torch.tensor([ont.index("MYB"), ont.index("FGC"), ont.index("EBO")])
    mat = torch.arange(3 * ont.num_chains, dtype=torch.float).view(3, -1)
    chain_of = torch.as_tensor(ont.chain_of)
    pi_of = torch.as_tensor(ont.pi_of, dtype=torch.float).nan_to_num(0)
    s, pi, q = gather_on_chain(mat, y, chain_of, pi_of)
    assert len(s) == 2 and q.tolist() == [ont.chains.index("neutrophil"), ont.chains.index("erythroid")]
    assert pi.tolist() == [0.25, 1.0]


def test_ramp():
    assert ramp(1.0, 0.5, start=1, length=2) == 0
    assert ramp(1.0, 2.0, start=1, length=2) == 0.5
    assert ramp(1.0, 9.0, start=1, length=2) == 1.0
