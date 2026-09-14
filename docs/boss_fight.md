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

## The run-back: route execution, not pathfinding

The way from the bonfire to the fog gate never changes. That makes it a
*replay* problem, not a navigation problem — no map, no localisation, no
search. What it does need is noticing when the replay has gone wrong, because a
run-back that silently walked somewhere else turns every following attempt into
noise.

Write the rough legs as YAML:

```yaml
name: undead asylum -> asylum demon
waypoints:
  - {name: out of the bonfire room, buttons: [forward], seconds: 4.0}
  - {name: turn to the stairs,      buttons: [forward, right], seconds: 1.0}
  - {name: up the stairs,           buttons: [forward], seconds: 3.5}
  - {name: through the fog,         buttons: [forward], seconds: 2.0}
```

Stand where the run-back starts, then walk it once for real:

```
flyds1 route --steps route.yaml --record route.npz --source mss
flyds1 --set env.boss.route_path=route.npz --set env.kind=boss live
```

Recording captures a reference frame at the end of every leg. On later runs
each leg is checked against its reference (zero-mean normalised correlation on
a 24x42 greyscale patch — coarse on purpose, so torch flicker and the
character's own animation do not count as "somewhere else"). A leg that does
not match gets an unstick wiggle and one more try; if it still does not match,
the attempt is abandoned and counted in `routes_lost` rather than continued.

`flyds1 route --verify route.npz` walks it and prints the match score per leg,
which is how you tune `--threshold`: scores on a good run tell you how much
headroom you have.

**Local steering can come from the brain.** `Route.run` takes a `steer`
callback returning extra buttons per frame, so the route supplies the direction
while the connectome's looming-sensitive cells keep the character off the
walls. That division — route memory plus reactive avoidance — is how insects
navigate; they do not plan paths either.

## Would it be better to read the game's memory instead?

For the *bookkeeping*, yes, clearly: player HP, boss HP, position and animation
state as exact numbers instead of pixel heuristics removes every calibration
problem on this page at once, and real coordinates would make actual
pathfinding trivial. That is what soulsgym does for DS3, and the memory offsets
for DS1 Remastered are documented by the community. The costs are version-
specific offsets to maintain, and that anti-cheat treats process memory access
as tampering — fine for an offline single-player session, not for going online
afterwards.

For the *experiment*, no. The question here is whether a fly's visual system can
drive behaviour in a game. A fly handed exact coordinates is not a fly playing
Dark Souls; it is a lookup table. The vision has to stay pixels.

So the sensible split is hybrid: vision from pixels, bookkeeping from memory if
you want the robustness. The code is arranged for it —
:class:`flyds1.envs.detectors.FightStateDetector` is the whole state interface,
and a memory-backed implementation drops in without touching the env, the
retina, or the brain.

## Alt-tabbing out is safe by default

Reported live: tabbing out of Dark Souls left the agent clicking and moving the
mouse across the desktop. `SendInput` on Windows (and, in practice, X11
synthetic events) deliver to whatever window currently has OS focus, not to a
window you name -- there is no way to say "send this specifically to Dark
Souls" at that API level. Every real backend now checks the foreground
window's title before acting and releases whatever it was holding the instant
focus is lost, so a key never reads as stuck down when you tab back in.

This is on by default: `window_name` defaults to `"DARK SOULS"`, a
case-insensitive substring match against the title bar (matches "DARK
SOULS™: REMASTERED"). Override it for a different game with
`--set env.game.window_name=...` (or `env.boss.window_name`); setting it to
`None` disables the guard entirely, which is never the right call for normal
use.

## Dying past enemies on the route is expected, not a failure

A route that runs past live enemies -- skeletons on the way to a boss -- will
occasionally get the character killed. Dark Souls' own answer to that is to
respawn at the last bonfire, which for a run-back is the route's own starting
point, so the fix is not "teach the agent to dodge or fight" (a much larger,
unproven claim -- see `docs/results.md` on how little steering information this
pipeline has been shown to extract even from a plain visual target) but
"recognise the death screen, wait out the reload, start the route over."
`Route.run` does this by default (`handle_death=True`), up to
`route_max_respawns` times (default 3, configured via `env.boss.*`) before
giving up on the attempt as `died_out` rather than looping forever.

The practical recipe for a route through a hazard like the graveyard skeletons
on the way to Pinwheel: include `sprint_roll` in the relevant waypoints'
buttons so the character runs rather than walks, do not script any fighting,
and let the occasional death be absorbed by the respawn-and-restart above --
this is also the standard human strategy at that point in the game.
