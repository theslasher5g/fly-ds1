import numpy as np
import pytest

from flyds1.connectome.codex import _parse_id, load_codex_dump, read_neuron_csv
from flyds1.connectome.graph import build_wired_network
from flyds1.connectome.loader import load_connectome
from flyds1.connectome.schema import NT_SIGNS, NeuronTable, SynapseTable
from flyds1.connectome.synthetic import SyntheticConfig, build_synthetic_connectome


def test_synthetic_has_the_required_annotations(connectome):
    assert connectome.neurons.photoreceptor_mask().sum() > 0
    assert connectome.neurons.descending_mask().sum() > 0
    assert connectome.n_edges > 0
    # photoreceptors must carry retinotopy or step 3 cannot work
    coords = connectome.neurons.eye_coord[connectome.neurons.photoreceptor_mask()]
    assert np.isfinite(coords).all()


def test_synthetic_is_deterministic():
    a = build_synthetic_connectome(SyntheticConfig(rings=2, seed=7))
    b = build_synthetic_connectome(SyntheticConfig(rings=2, seed=7))
    assert np.array_equal(a.synapses.counts, b.synapses.counts)
    c = build_synthetic_connectome(SyntheticConfig(rings=2, seed=8))
    assert not np.array_equal(a.synapses.counts, c.synapses.counts)


def test_dales_law_holds(network):
    """Every neuron's outgoing weights share one sign."""
    W = network.W.tocsc()
    for pre in range(W.shape[1]):
        column = W.data[W.indptr[pre] : W.indptr[pre + 1]]
        if len(column):
            assert (column > 0).all() or (column < 0).all(), f"neuron {pre} is both"


def test_signs_follow_the_neurotransmitter(connectome):
    net = build_wired_network(connectome, min_synapse_count=0)
    W = net.W.tocsc()
    signs = net.neurons.signs()
    for pre in range(W.shape[1]):
        column = W.data[W.indptr[pre] : W.indptr[pre + 1]]
        if len(column) and signs[pre] != 0:
            assert np.sign(column[0]) == np.sign(signs[pre])


def test_weak_edges_are_filtered(connectome):
    strong = build_wired_network(connectome, min_synapse_count=50)
    weak = build_wired_network(connectome, min_synapse_count=0)
    assert strong.W.nnz < weak.W.nnz


def test_modulatory_neurons_carry_no_weight():
    neurons = NeuronTable(
        ids=[1, 2],
        cell_type=["a", "b"],
        super_class=["central", "central"],
        nt_type=["dopamine", "acetylcholine"],
        side=["left", "left"],
        eye_coord=np.full((2, 2), np.nan),
    )
    synapses = SynapseTable([1, 2], [2, 1], [100.0, 100.0])
    from flyds1.connectome.schema import Connectome

    net = build_wired_network(Connectome(neurons, synapses), drop_modulatory=True)
    assert net.W.nnz == 1  # only the cholinergic edge survives
    assert NT_SIGNS["dopamine"] == 0.0


def test_root_ids_survive_parsing():
    """FlyWire root ids exceed float64 integer precision."""
    big = 720575940600000001
    assert _parse_id(str(big)) == big
    assert _parse_id(f"{big}.0") == big
    assert int(float(str(big))) != big  # the bug this guards against


def test_codex_round_trip(tmp_path):
    (tmp_path / "neurons.csv").write_text(
        "root_id,cell_type,super_class,nt_type,side,hex1,hex2\n"
        "720575940600000001,R1-6,sensory,HIS,right,0,0\n"
        "720575940600000002,R1-6,sensory,his,right,1,0\n"
        "720575940600000003,L1,optic,glut,right,,\n"
        "720575940600000004,DNa02,descending,ACH,right,,\n"
    )
    (tmp_path / "connections.csv").write_text(
        "pre_root_id,post_root_id,neuropil,syn_count\n"
        "720575940600000001,720575940600000003,ME_R,30\n"
        "720575940600000001,720575940600000003,LO_R,12\n"
        "720575940600000003,720575940600000004,CB,9\n"
        "720575940600000009,720575940600000004,CB,5\n"  # unknown neuron: dropped
    )
    c = load_codex_dump(tmp_path)
    assert len(c.neurons) == 4
    assert c.meta["dropped_unannotated_edges"] == 1
    # the two neuropil rows of the same pair are summed
    assert c.synapses.counts.max() == 42
    assert c.neurons.photoreceptor_mask().sum() == 2
    assert c.neurons.descending_mask().sum() == 1


def test_codex_reports_missing_columns(tmp_path):
    path = tmp_path / "neurons.csv"
    path.write_text("id,label\n1,x\n")
    with pytest.raises(KeyError, match="cell_type"):
        read_neuron_csv(path)


def test_loader_spec_parsing():
    c = load_connectome("synthetic:rings=1,seed=2")
    assert c.source == "synthetic"
    assert c.meta["config"]["rings"] == 1
    with pytest.raises(ValueError):
        load_connectome("nonsense:x")


def test_network_save_load(network, tmp_path):
    from flyds1.connectome.graph import WiredNetwork

    path = network.save(tmp_path / "net.npz")
    back = WiredNetwork.load(path)
    assert back.n_neurons == network.n_neurons
    assert (back.W != network.W).nnz == 0
    assert np.array_equal(back.output_index, network.output_index)


def test_central_and_descending_scale_with_eye_size():
    """Regression: n_central and n_descending were pinned at 200/24 regardless
    of rings, so a rings=15 eye (Drosophila's real ~750 ommatidia/eye) got a
    central brain and a motor bottleneck sized for a much smaller eye --
    0.7% central brain, 0.09% descending, instead of a proportion that keeps
    pace with the eye."""
    small = SyntheticConfig(rings=6)
    large = SyntheticConfig(rings=15)
    assert small.n_central == 200 and small.n_descending == 24  # unchanged at the old baseline
    assert large.n_central > small.n_central * 4
    assert large.n_descending > small.n_descending * 4

    # explicit overrides (as test fixtures use) are still respected
    explicit = SyntheticConfig(rings=15, n_central=60, n_descending=8)
    assert explicit.n_central == 60 and explicit.n_descending == 8
