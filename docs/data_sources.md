# Getting a real connectome

The pipeline is written against two tables — neurons and connections — so any
dataset that can produce those two works. `flyds1 fetch` prints the short
version; this is the long one.

## FlyWire / FAFB (whole adult female brain)

1. Open <https://codex.flywire.ai/>, sign in, accept the data terms.
2. Downloads → pick a snapshot (783 is the usual public one) and take:
   * the neuron annotation CSV (`root_id`, `cell_type`, `super_class`,
     `nt_type`, `side`, …)
   * the connection CSV (`pre_root_id`, `post_root_id`, `syn_count`, often one
     row per neuropil — the loader aggregates them)
3. Put both in one directory and point the config at it:

```bash
flyds1 build --set connectome.spec=codex:data/flywire_783 --out data/net.npz
flyds1 tune  --set connectome.spec=codex:data/flywire_783   # pick the gain
```

Scale: ~140k neurons, ~2.7M edges above the 5-synapse threshold. The sparse
matrix is fine; a full-brain PPO rollout is not cheap — start from a subgraph
(optic lobe + central + descending) while prototyping.

**Citation and licence.** FlyWire data carry citation requirements
(Dorkenwald et al. 2024, *Nature*; Schlegel et al. 2024, *Nature*) and the
snapshot's own licence terms. Check them before publishing anything built on
this; the loader records the citation string in `connectome.meta`.

## Male CNS / MANC (Janelia + Google Research)

Same two-table shape, different spelling (`bodyId`, `type`, `class`, `weight`).
Those aliases are already in `NEURON_COLUMNS` / `CONNECTION_COLUMNS`, so the
same loader reads them. Browse the volume in Neuroglancer to find the body ids
you care about; export connectivity as CSV and use `connectome.spec=codex:<dir>`
(the loader is named after Codex but is not specific to it).

The male CNS volume includes the ventral nerve cord, which means descending
neurons have real *targets* rather than just being a read-out — worth it if you
ever want to model the body rather than press keys.

## natverse (live queries)

```bash
pip install -e ".[connectome]"
python -c "import fafbseg; fafbseg.flywire.set_chunkedgraph_secret('<CAVE token>')"
```

```python
from flyds1.connectome.natverse import connectome_from_fafbseg
connectome = connectome_from_fafbseg(root_ids=my_ids, dataset="public")
```

Use this when you want a specific subgraph without downloading a full snapshot.
`connectome_from_frames` takes the two pandas frames directly, so anything you
can get into that shape (an R natverse export, a cached parquet file) works too.

## What the loader needs to find

| Field | Why | If missing |
|---|---|---|
| `super_class` = `sensory` + eye coordinates | step 3 injects the screen here | no vision; falls back to nothing, `RetinaFrontEnd` raises |
| `super_class` = `descending` (or cell types containing `dn`) | step 5 reads the motor output here | `FlyBrainExtractor` raises — there is nothing to decode |
| `nt_type` | the sign of every synapse | neurons default to sign 0 and their edges are dropped |
| `hex1`/`hex2` **or** soma positions | retinotopy of the photoreceptors | `retinotopy.py` raises rather than inventing coordinates |

Photoreceptor retinotopy is the one that most often needs work: FlyWire's main
annotation table does not carry hex column ids, so either join the optic-lobe
column annotations onto it, or let `eye_coords_from_soma` fit a plane through the
somata (neighbourhood-preserving, not a calibrated optical-axis map).
