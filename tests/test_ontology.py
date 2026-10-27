import numpy as np


def test_structure(ont):
    assert ont.num_fine == 21 and ont.num_chains == 3
    neu = [ont.index(c) for c in ["PMO", "MYB", "MMZ", "NGB", "NGS"]]
    np.testing.assert_allclose(ont.pi_of[neu], [0, 0.25, 0.5, 0.75, 1.0])
    assert ont.chain_of[ont.index("FGC")] == -1 and np.isnan(ont.pi_of[ont.index("FGC")])
    assert ont.lineage_of[ont.index("FGC")] == ont.lineage_of[ont.index("PMO")]
    assert ont.lineage_mask().sum() == ont.num_fine


def test_tree_distance_ordering(ont):
    D = ont.tree_distance()
    i = ont.index
    assert np.all(np.diag(D) == 0)
    adjacent = D[i("MYB"), i("PMO")]
    distant = D[i("NGS"), i("PMO")]
    off_chain = D[i("FGC"), i("PMO")]
    cross = D[i("EBO"), i("PMO")]
    assert 0 < adjacent < distant <= off_chain < cross
    # a full slip on a short chain is never worse than an off-chain confusion
    assert D[i("EBO"), i("PEB")] <= off_chain
