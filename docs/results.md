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

### The first run did not learn, and the controls said why

60k steps on the chase task, evaluated deterministically every 10k: every point
stayed within noise of the untrained network (returns −13 to −16 against an
untrained −13.1 ± 13.9, death rate wandering between 38% and 56% with a
standard error of ±12 points). Rather than tune hyper-parameters against that,
two controls were run on the *same* task and budget:

| control | sees | result |
|---|---|---|
| MLP, privileged | ground-truth enemy distance and bearing | −11.1 → −11.5 over 150k steps, critic `explained_variance` 0.99 |
| MLP, retina | exactly what the fly sees | −14.9 → −13.5 over 150k steps |
| scripted oracle | ground truth, 6 lines of code | **−4.0 ± 2.0** |

An unconstrained MLP with perfect information and a near-perfect critic could
not approach a six-line script. **That rules out the fly pipeline as the
cause** — whatever was blocking learning blocked everything equally. Three
things turned out to be wrong, in increasing order of importance.

### 1. The action space was noise

Buttons were a continuous vector thresholded at zero. Measured on an untrained
policy: the action means have magnitude **0.004** while the policy's own
standard deviation is **1.0**. During training each button therefore flipped
with probability ≈0.47 regardless of what had been learnt, and the
deterministic evaluation decided behaviour by the *sign* of numbers 250× below
that noise. Held buttons are now `MultiBinary` with a Bernoulli policy, where a
logit shift maps straight onto a change in behaviour.

### 2. The critic was squeezed through the bottleneck

With a shared features extractor the value function saw the same 8–24
descending neurons as the policy. The critic is not part of the animal, so the
handicap bought nothing; `FlyActorCriticPolicy` gives it the raw observation
while the actor still goes through the connectome.

Together these two moved the fly's `approx_kl` per update from `2.5e-6` to
`~1e-3` — the policy went from effectively frozen to actually moving.

### 3. The task had almost no controllable range

The deeper problem: on the chase task even the scripted oracle loses ~40 hp per
episode, and most of the return is decided by where the enemy spawns. A
benchmark whose optimum is barely distinguishable from doing nothing cannot
teach a policy or measure one.

The target-fixation task was added for that reason — steer at a bright marker,
which relocates when reached:

| policy | return | markers reached |
|---|---|---|
| standing still | +6.00 ± 0.00 | 0.0 |
| walking blindly forward | −6.09 ± 1.76 | 0.1 |
| forward while turning | +5.99 ± 0.19 | 0.0 |
| **scripted oracle** | **+40.51 ± 1.04** | **2.6** |

A 34-point spread with a standard error near 1, against 11 points of spread
with ±3 on the chase task. It is also the behaviour the connectome is for:
visual fixation is what T4/T5 and the lobula plate do.

### Learning the target task

PPO, 30 held-out episodes evaluated deterministically every 20k steps:

| steps | MLP on ground truth | scripted oracle |
|---|---|---|
| random actions | +6.4 ± 0.1 (0.0 markers) | |
| untrained | +5.6 ± 0.4 (0.0) | |
| 20k | +37.2 ± 0.7 (2.1) | |
| 40k | +40.3 ± 0.7 (2.5) | **+40.5 ± 1.0 (2.6)** |
| 60k | +40.8 ± 0.6 (2.5) | |

The control reaches the scripted oracle's score in 40k steps. That is the
confirmation that mattered: with a benchmark that has range and an action space
that means something, the same PPO setup that looked broken learns immediately.

The fly, on the same task with everything the measurements asked for — a retina
covering the whole screen (rings=6, 254 photoreceptors), the brain kept warm
across the episode, `MultiBinary` actions, the critic on the observation:

| steps | return | markers |
|---|---|---|
| random actions | +6.4 ± 0.3 | 0.0 |
| untrained | +1.9 ± 0.5 | 0.0 |
| 20k | +4.8 ± 0.6 | 0.0 |
| 40k | +4.3 ± 1.0 | 0.1 |
| 60k | +4.5 ± 0.7 | 0.0 |
| 80k | +6.0 ± 0.0 | 0.0 |
| 100k | −0.5 ± 1.5 | 0.0 |

It never beats standing still (+6.0) and never reaches a marker, while the
control was at oracle level after 40k. The difference is not the algorithm, the
budget or the task — those are shared — it is what arrives at the descending
neurons.

### Why, and how far the signal gets

The fly did not follow. Probing why produced the most useful numbers in this
document, so they are worth stating in order.

**The descending neurons are not told in time.** Drive the retina with a target
drifting on the left versus the right of the screen and watch the descending
population separate:

| simulated time | ‖DN_left − DN_right‖ |
|---|---|
| 2 frames (66 ms) | 0.00000 |
| **4 frames (132 ms)** — the default `stack_k` | **0.00025** |
| 8 frames (264 ms) | 0.037 |
| 16 frames (528 ms) | 0.142 |

The path from photoreceptor to descending neuron is six or more synapses of
10–50 ms each. Unrolling the network from rest over a 4-frame stack asks it to
answer before the signal has arrived. Either use a much longer stack (4× the
compute per policy step) or run the brain in the environment
(`brain_location: wrapper`), where it is never cold.

**The eye was pointed at a third of the screen.** `coverage()` said 100%, which
was true and useless: every facet was on screen, but a 3-ring eye spans ±15°
and the monitor spans 90°, so the retina sampled 58% of its width and the
marker spent most of the episode outside the visual field. `screen_coverage()`
now measures the direction that matters and `selftest` fails on it.

**A random central brain is a blender.** Even warm, even with full coverage,
the modulation a target position produces decays through the hierarchy:

| population | modulation depth (std across target positions / mean rate) |
|---|---|
| photoreceptors | 2.0 |
| optic lobe | 0.042 |
| visual projection | 0.016 |
| central brain | 0.009 |
| descending | 0.010 |

A linear probe trained on the descending rates to predict a steering oracle's
actions lands at the majority-class baseline (0.88–0.89 against 0.88), while
the same probe on the retinal input reaches 0.93. Two flaws in the generator
were found and fixed this way — lobula cells with randomly scattered receptive
fields instead of tiled ones, and descending neurons reachable only through the
random central brain — and each fix helped a little (the correlation between a
DN's preferred azimuth and the stimulus that drives it is |0.68| in a clean lab
stimulus), but not enough to make steering decodable inside the cluttered arena.

**That is a statement about the generator, not about Drosophila.** The
synthetic connectome was built to exercise the plumbing, and it does: every
stage is verified, and the numbers above were all measured through it. But its
central brain is random wiring, and random wiring does not preserve where
things are. The animal has what this stand-in lacks — small-object detectors
that separate figure from background, retinotopic central projections, strong
identified LC→DN pathways. **This is the point where the real connectome stops
being optional.**

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

## Two more fixes from testing against real Dark Souls footage

**Central brain and descending neurons did not scale with eye size.** Moving
the default eye to `rings=15` (Drosophila's real ~750 ommatidia/eye, see
above) exposed that `n_central` and `n_descending` had stayed pinned at 200 and
24 -- numbers sized for the old 127-facet default. At the new eye size that is
97% optic lobe, 0.7% central brain, 0.09% descending: a caricature, not a
smaller fly. Both now scale with the eye (preserving the ratio they had at the
old rings=6 default), giving 1135 central neurons and 136 descending neurons at
rings=15 -- a much less absurd proportion, at a cost of ~1s extra build time.

**`--set env.boss.*` silently did nothing under `env.kind=game`.** The game and
boss environments each have their own config section
(`env.game.frame_source` vs `env.boss.frame_source`, etc.), and setting the
wrong one produced no error -- the environment just kept using its own
section's default, which for the untouched one is a dummy synthetic source.
Diagnosed live: a real Dark Souls session on screen, and the fly's "game" panel
showing a synthetic grating instead. `flyds1` now warns on stderr whenever a
`--set` targets `env.game.*` or `env.boss.*` while `env.kind` is set to the
other one.
