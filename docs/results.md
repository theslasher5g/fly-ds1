# What has actually been measured

Everything here was produced by the code in this repository on the **synthetic**
connectome. Nothing on this page is a statement about *Drosophila*; it is a
statement about whether the machinery works.

Setup unless stated otherwise: `synthetic:rings=3` (1674 neurons, 8526 edges
after the 5-synapse threshold, 74 photoreceptors per eye, 8 descending
neurons), arena at 160×90, `motion_mode: pooled`, `stack_k: 4`, CPU only.

## Step 2: the network computes something

**Backend agreement.** numpy reference vs PyTorch over 20 frames of identical
drive: max absolute difference `8.9e-16` (float64). Guarded by
`tests/test_net.py::test_backends_agree` at `1e-10`.

**The bias is not optional.** With `bias=0` every rate settles at exactly zero,
whatever the stimulus — photoreceptors are histaminergic, so with rectified
rates there is nothing for them to inhibit. This is asserted, not just noted:
`test_zero_bias_would_silence_the_network`.

**The gain decides whether a signal survives the hierarchy.** `flyds1 tune`
drives the eye with random patterns and reports how much the rates differ
between stimuli at each stage:

| gain | spectral radius | DN mean rate | DN std across stimuli | peak rate |
|---|---|---|---|---|
| 0.5 | 0.149 | 1.23 | 1.4e-04 | 2.0 |
| 1.0 | 0.298 | 1.60 | 2.6e-03 | 3.0 |
| 2.0 | 0.596 | 2.89 | 2.8e-02 | 7.0 |
| **3.0** | **0.894** | **5.87** | **1.2e-01** | **20.2** |
| 4.0 | 1.191 | 14.67 | 3.8e-01 | 65.5 |

At gain 1.0 the stimulus-driven variance falls from `5.7e-01` at the
photoreceptors to `8.8e-02` at the central brain and `2.6e-03` at the descending
neurons — a 200× attenuation that leaves the policy looking at a constant. The
default is 3.0: the strongest signal that still has a spectral radius below 1.
Gain 4.0 transmits more but is super-critical, where "responsive" and "runaway"
are hard to tell apart.

## Step 3: the eye is direction-selective

A drifting sine grating, sampled through the ommatidia and correlated:

| stimulus | mean axis-0 response |
|---|---|
| +120 px/s (rightward) | `+7.3e-03` |
| −120 px/s (leftward) | `−7.4e-03` |
| static | `0.0` (exactly) |

Sign flips with direction, magnitude matches, no response without motion.
`tests/test_vision.py` asserts all three.

## Step 6: PPO runs, and what it does

**Throughput.** ~105 environment steps/s with the brain in the policy
(`stack_k=4`, so 4×4 = 16 Euler substeps of a 1674-neuron network per step, on
CPU). The arena alone runs at ~1900 steps/s, so the brain is the cost.

**The policy moves.** `approx_kl` ≈ `5e-5` per update against ≈ `2e-4` for a
plain 64×64 MLP on the same observations. The ~4× gap is the deliberately
linear decoder reading 8–24 descending neurons; it is a constraint, not a bug.

**Learning in the arena.** 60k steps of PPO, evaluated deterministically over
16 held-out episodes (fixed seeds) every 10k steps:

| | return | mean length | died | critic `explained_variance` |
|---|---|---|---|---|
| random actions | −13.4 ± 14.3 | 258 | 56% | — |
| untrained fly | −13.1 ± 13.9 | 271 | 56% | — |
| 10k steps | −16.5 ± 11.9 | 269 | 44% | +0.38 |
| 20k steps | −14.4 ± 13.9 | 271 | 38% | +0.52 |
| 30k steps | −16.5 ± 15.3 | 261 | 56% | +0.31 |
| 40k steps | −15.1 ± 14.5 | 265 | 50% | +0.67 |
| 50k steps | −13.6 ± 14.3 | 269 | 44% | +0.55 |
| 60k steps | −14.8 ± 13.8 | 263 | 56% | −0.33 |

**Read this as: no learning.** With 16 episodes and a return spread of ~14, the
standard error is ±3.5, and the death rate's is ±12 percentage points — every
number in that table is within noise of the untrained network. The dip in death
rate at 20k is not evidence of anything. What *is* solid: the loop runs, the
gradients arrive, and the critic mostly does better than predicting the mean
(which it never did before the reward normalisation).

Where the next effort probably belongs, in order:

1. **Budget.** 60k steps is small for a vision-driven PPO task; a million is a
   more usual order of magnitude. At 105 steps/s that is ~2.6 hours on CPU.
2. **Give the critic more than the bottleneck.** With
   `share_features_extractor=True` the value function sees the same 8–24
   descending neurons as the policy. There is no reason to handicap the critic
   — it is not part of the animal — but feeding it the raw observation needs a
   custom SB3 policy, since `MlpPolicy` only ever hands it a features
   extractor. This is the cheapest structural change on this list.
3. **Memory.** `stack_k=4` is 130 ms of history. Escaping something that
   approaches over several seconds may simply need more, i.e. a larger `k` or
   `RecurrentPPO`.
4. **A wider motor channel.** This configuration has 8 descending neurons; a
   larger connectome has more, and the decoder is linear, so the channel width
   is a hard ceiling on what the policy can express.
5. **A denser reward.** Damage is sparse and delayed. Rewarding distance kept
   from the enemy every step would give the policy gradient something to hold
   on to — at the cost of telling the agent the answer instead of letting it
   see it.

## Two measurement mistakes worth recording

**An arena that did not need eyes.** The first reward weighted distance
travelled at 1.0 per tile. That made "hold back and turn" worth ~26 per episode
while dying cost 5, so the optimal policy was a constant motor command and
training *degraded* an accidentally-good initialisation (+23.8 → +1.4 over 30k
steps). A visual model cannot be evaluated on a task solvable with the eyes
shut; survival and damage now dominate, and the enemy is slower than the player
so escaping it requires seeing it.

**A critic that was worse than a constant.** `explained_variance` was negative
for the entire first run — the value function was less accurate than predicting
the mean return, which makes every advantage noise and every policy update a
random walk. Normalising the reward (`VecNormalize(norm_reward=True)`) fixed it.

## Reproducing

```bash
pip install -e ".[torch,rl,dev]"
pytest -q                      # 72 tests, ~20 s
flyds1 selftest                # numeric checks on every stage
flyds1 tune                    # the gain table above
flyds1 train --timesteps 60000 # the learning run
```
