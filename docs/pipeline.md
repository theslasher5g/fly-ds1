# How the parts fit together

```
             ┌──────────────── step 1 ────────────────┐
  Codex CSV ─┤ codex.py / natverse.py / synthetic.py  ├─> Connectome
             └────────────────────────────────────────┘   (NeuronTable, SynapseTable)
                                │ step 2a: graph.py
                                v
                        WiredNetwork  (W[post,pre] signed sparse, tau, input/output index)
                                │
        ┌───────────────────────┴───────────────────────┐
        │ step 2b                                       │
   net/reference.py (numpy, no grad)          net/rate_rnn.py (torch, frozen W)
        │                                               │
        └──────────── used by ──────────────┬───────────┘
                                            │
   frame ──> vision/frontend.py ──────> observation ──> vision/encoder.py ──> drive
   (step 3: ommatidia.py + reichardt.py)   (fixed)        (trainable)
                                                              │
                                                              v
                                                        ConnectomeRNN
                                                              │ descending rates
                                                              v
                                          motor/decoder.py ── action ──> motor/actions.py
                                                                             │
                                                     envs/game.py ───────────┘ keys/mouse
```

## Who owns what

**Fixed physics (environment side).** Optics and elementary motion detection are
not parameters, so they run in `RetinaWrapper` and their output *is* the
observation. Drop the wrapper and you get raw frames back — useful when you want
to look at what the agent looks at.

**Trainable (policy side).** The encoder (gain/offset per photoreceptor, gain per
T4/T5 subtype), the linear decoder, and optionally a declared number of plastic
edges and the resting bias. Everything else about the network comes from the
connectome and stays there.

## The timestep, three ways

| Symbol | Default | Meaning |
|---|---|---|
| `rate.dt` | 4.17 ms | Euler step of the neural dynamics |
| `rate.steps_per_frame` | 4 | substeps per environment frame → 16.7 ms of simulated time |
| env `dt` | 33 ms (arena) / `1/target_fps` (game) | wall-clock interval the retina integrates over |

`dt` must stay below the smallest membrane time constant or forward Euler
oscillates; `check_timestep` enforces it rather than letting you discover it as
"interesting dynamics". The frame interval is what the Reichardt detectors and
the light adaptation use, so it must match the environment's real rate — that is
why `RetinaWrapper` infers it from the env instead of taking a default.

## Failure modes worth knowing

These are all things that produced a plausible-looking but dead pipeline during
development, and each now has a test:

1. **Silent network.** Rectified rates + histaminergic photoreceptors + zero
   bias = every rate zero forever. → `net/bias.py`.
2. **Signal death in depth.** Weights normalised per input budget attenuate at
   every layer; by the descending neurons the stimulus is gone. → `flyds1 tune`,
   `net/diagnostics.py`.
3. **Features collapsed by the standardiser.** Estimating variance within a
   batch of near-simultaneous frames underestimates the temporal variance by
   orders of magnitude; a unit-variance prior with a full sample's weight does
   the same for hundreds of updates; normalising with statistics that include
   the current sample returns exactly zero for a batch of one. All three
   silently zero the gradient. → `motor/decoder.py:RunningStandardiser`.
4. **Root ids through float64.** FlyWire ids are ~7.2e17 and do not fit in a
   float64 mantissa: `int(float(id))` merges distinct neurons. → `_parse_id`.
5. **Stateful net + shuffled minibatches.** PPO replays observations out of
   order; a network carrying hidden state across steps computes a different
   thing at replay than at collection. → the two `brain_location` modes.
6. **A featureless world.** Flat-shaded walls give the motion detectors nothing
   to correlate. The arena's walls are textured for that reason, and a test
   asserts the frames actually vary.

## Docking onto an existing Dark Souls gym environment

Community projects already wrap the game (`soulsgym` and similar). Nothing here
needs to duplicate that work: `RetinaWrapper` takes *any* Gymnasium env whose
observation is a frame, so docking is

```python
env = ThirdPartySoulsEnv(...)            # produces (H, W, 3) frames
env = RetinaWrapper(env, network, FrontEndConfig(), dt=1 / 30)
env = ObsStackWrapper(env, k=6)          # if the brain runs in the policy
model = PPO("MlpPolicy", env, **fly_policy_kwargs(network, env.layout))
```

Two things to check when you do:

* **Observation shape.** If their env returns something other than a frame
  (feature vectors, stacked greyscale), take the frame out of `info` or use
  their raw-pixel option; the retina needs pixels.
* **Action space.** Theirs is probably `Discrete` or `MultiDiscrete`, while the
  decoder emits a continuous vector. Either map the decoder's output through
  `ActionSpec.decode` into their indices, or give their env a `Box` adapter —
  a ten-line wrapper either way. Keep the decoder linear whichever you pick.

Their reward is usually better than the HUD-pixel estimate in `envs/reward.py`
(they read the game's memory), so prefer it when it is available.

## Extending it

* **A different game**: write the key bindings as an `ActionSpec`, the HUD
  regions as `BarRegion`s, and point `GameConfig` at the window. No code change.
* **Spiking instead of rates**: `net/spiking.py` builds a Brian2 LIF network
  from the same `WiredNetwork` — same graph, same signs, same time constants.
  What it has to invent (threshold, reset, refractory period, mV per unit
  weight) sits in `LIFParameters` with comments saying so. Its tests skip where
  Brian2 does not import: Brian2 currently needs numpy<2.
* **More brain regions**: nothing in the pipeline is optic-lobe specific. Add
  the neurons to the connectome and they are in the network; only the input and
  output index maps care about annotations.
