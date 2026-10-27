import numpy as np

from hemahier.decode import product_decode
from hemahier.metrics import evaluate, maturity_quality, severity_breakdown


def test_severity_metrics(ont):
    i = ont.index
    y = np.array([i("PMO"), i("PMO"), i("PMO"), i("PMO"), i("PMO")])
    pred = np.array([i("PMO"), i("MYB"), i("NGS"), i("FGC"), i("EBO")])
    br = severity_breakdown(y, pred, ont)
    assert br == {"adjacent_stage": 20.0, "same_chain_distant": 20.0, "same_lineage_other": 20.0,
                  "cross_lineage": 20.0}
    m = evaluate(y, pred, ont, rare_classes=[])
    assert m["cle"] == 20.0
    assert m["td"] == 100 * np.mean([0, 0.125, 0.5, 0.5, 1.0])


def test_maturity_quality_perfect_order(ont):
    ids = ont.chain_members[0]
    y = np.repeat(ids, 10)
    mat = np.zeros((len(y), ont.num_chains))
    mat[:, 0] = ont.pi_of[y] + np.random.default_rng(0).uniform(0, 0.01, len(y))
    q = maturity_quality(y, mat, ont)
    # rho < 1 even for perfect order because pi is tied within a stage; order accuracy is exact
    assert q["mat_rho"] > 0.95 and q["ord_acc"] == 100.0


def test_product_decode_keeps_chain_mass(ont):
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(ont.num_fine), size=4)
    mat = rng.uniform(size=(4, ont.num_chains))
    out = product_decode(p, mat, ont)
    np.testing.assert_allclose(out.sum(1), 1.0)
    for ids in ont.chain_members:
        np.testing.assert_allclose(out[:, ids].sum(1), p[:, ids].sum(1))
