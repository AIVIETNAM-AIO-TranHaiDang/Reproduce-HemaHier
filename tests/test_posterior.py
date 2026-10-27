import torch

from hemahier.models.heads.hemahier import HemaHierHead, coupled_log_posterior


def test_posterior_is_distribution_and_marginal_matches_lineage(ont):
    torch.manual_seed(0)
    B = 16
    lin = torch.randn(B, ont.num_lineages)
    raw = torch.randn(B, ont.num_fine)
    mask = ont.lineage_mask()
    lineage_of = torch.as_tensor(ont.lineage_of)
    p = coupled_log_posterior(lin, raw, mask, lineage_of).exp()
    torch.testing.assert_close(p.sum(1), torch.ones(B))
    # Eq. 2 factorization: summing p_f over a lineage gives back p_c exactly
    torch.testing.assert_close(p @ mask.float().T, lin.softmax(-1))


def test_head_shapes_params_and_loss(ont):
    head = HemaHierHead(ont, 384, 384)
    n = sum(p.numel() for p in head.parameters())
    n_mat = sum(p.numel() for m in (head.chain_emb, head.maturity_head) for p in m.parameters())
    assert 4.5e6 < n < 5.5e6, n          # paper: 5.0M head parameters
    assert n_mat < 0.25e6, n_mat         # paper: maturity components < 0.25M
    out = head(torch.randn(8, 384), torch.randn(8, 384))
    assert out["log_probs"].shape == (8, ont.num_fine)
    assert out["maturity"].shape == (8, ont.num_chains)
    y = torch.tensor([ont.index(c) for c in ["PMO", "MYB", "NGS", "FGC", "EBO", "PEB", "ART", "LYT"]])
    loss, logs = head.loss(out, y, epoch=10.0)
    loss.backward()
    assert torch.isfinite(loss) and {"joint", "raw", "lineage", "maturity", "rank"} <= set(logs)
    assert head.chain_emb.weight.grad is not None
