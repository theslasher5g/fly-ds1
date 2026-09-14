# Pointing the fly at a boss

The goal this supports: **one boss, many attempts, watched live**. Not a
playthrough — see the note at the end for why.

Everything below except the last two steps runs without the game.

## 1. Record a minute of footage

Capture the fight once yourself (any recorder), export frames, and put them in a
folder or a `.npy` stack of `(n, height, width, 3)` uint8.

```bash
flyds1 calibrate --frame shots/frame_0100.png --out calibration.png
```

It prints what the health-bar readers see and writes an annotated image: **red
boxes** are the HUD regions the reward reads, **green dots** are where the fly's
ommatidia look. Adjust until the boxes sit on the bars and the dots span the
frame. Two failure modes this catches in a minute each, which otherwise cost an
evening:

* boxes off the bars → the reward reads zero damage forever;
* dots clustered in the middle → the fly is looking at a third of your screen
  (the selftest fails on this too).

## 2. Check the fight detector on your footage

```bash
flyds1 watch --frames shots/ --png diagnostics/
```

Panels show frame, what the eye sees, and descending-neuron activity. What you
want to see: the DN bars *moving* when something happens on screen. If the
population barely moves, nothing downstream can act on it, and training will not
fix that — it is a perception problem, not a policy problem.

The fight detector needs its thresholds checked against your HUD:

| setting | meaning | how to check |
|---|---|---|
| `detectors.boss_hp` | where the boss bar is | `calibrate` overlay |
| `detectors.boss_bar_present` | bar counts as on screen above this | must stay **below** `boss_dead_below`, or the nearly-empty frames that prove a kill are thrown away (the config refuses otherwise) |
| `detectors.boss_dead_below` | lowest reading that counts as a kill | watch a real kill; a killing blow from well above this reads as a retreat |
| `detectors.death_*` | the "YOU DIED" screen | `watch` prints brightness and redness per frame |
| `detectors.confirm_frames` | frames a state must hold | raise it if cutscenes or hit flashes start phantom fights |

## 3. Record the run-back

After a death the game reloads at the bonfire and the agent has to walk back.
That is scripted, not learned:

```yaml
env:
  boss:
    runback:
      steps: [[["forward"], 4.0], [[], 0.5], [["forward", "sprint_roll"], 3.0]]
    enter_fog:
      steps: [[["forward"], 2.0]]
```

Buttons are names from the action spec; the second number is seconds. Time it
against your own route — a macro that ends up somewhere else turns every
attempt into a zero.

## 4. Look at the clock before you start

```bash
flyds1 budget
```

```
action_repeat 4 at 30 fps -> 133 ms per decision
attempt (max 900 decisions) plus run-back -> 142 s
25 attempts/hour, 22898 decisions/hour
1M decisions would take 44 hours (1.8 days) of real play
```

Published single-boss RL runs use millions of steps, and they have memory access
to the game state rather than pixel heuristics. `action_repeat` is the main
lever — and raising it is not only a speed trick: the descending neurons need
8–16 frames to hear about a stimulus at all ([results](results.md)), so
deciding every frame asks the brain to answer before it has been told.

## 5. Dry run, then live

```bash
# no input is sent; check the loop and the reward against the live screen
flyds1 live --set env.kind=boss --set env.boss.frame_source=mss --port 8000
```

Open the URL. Four panels: the game, what the eye sees, motion, and the
descending neurons that produce the key presses. The viewer is read-only and
runs in a browser on purpose — an overlay window on the same screen steals focus
from the game mid-fight.

When the dry run looks right, set `env.boss.dry_run: false` and
`env.boss.input_backend: auto`. **Single-player, offline, with the
`pydirectinput` failsafe (mouse into a screen corner) available.**

## 6. Keep the control run

If the fly ever kills the boss, that on its own means very little: the Asylum
Demon is the tutorial boss, and across hundreds of attempts a mashing policy
gets lucky eventually. Run the same number of attempts with random actions —
`flyds1 play --set env.kind=boss` with no `--model` — and compare kills per
attempt. Without that baseline there is no result, only an anecdote.

## What to expect

The watchable result is not the kill. It is the moment the boss winds up and the
looming-sensitive cells fire — visible in the DN panel, in real time, in a
single attempt. That is the circuit that saves a real fly from a swatter, doing
its job on a video game.

A kill would be a research result, and an unlikely one: pixel-derived reward, a
frozen connectome, a linear read-out of a few dozen descending neurons, and a
budget measured in days of real-time play. The infrastructure here is honest
about which of those it can help with — it removes the overhead and makes the
signal visible. It cannot make the signal exist.
