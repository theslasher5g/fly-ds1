# Plan: talking to the game directly, and watching the fly properly

Two requests, one document, because they share a spine: both want *more of the
game's and the brain's actual state*, instead of what can be inferred from
pixels and a 24-bar chart.

* Part 1 replaces screen heuristics and `SendInput` with the game's own memory.
* Part 2 replaces the server-rendered PNG filmstrip with a real instrument
  panel: per-hemisphere activity, population flow, cell-type maps, DN→key
  wiring.

Read Part 1's "What this cannot be" section before anything else. It is the
part that decides the ordering.

---

## Part 1 — the game's memory instead of its pixels

### Why bother

Everything currently fragile in `flyds1` is fragile because it reads the
screen and guesses:

| Today (pixels)                                  | Fails when                                   | With memory                       |
| ----------------------------------------------- | -------------------------------------------- | --------------------------------- |
| `DeathScreenDetector` — dark frame + red tint   | dark rooms, the fog gate, a red-lit corridor | a death/respawn counter, exact    |
| `FightStateDetector` — boss bar pixel fraction  | UI scale, resolution, HUD toggled off        | boss HP as an integer             |
| reward from HP-bar brightness                   | blood splatter, torch light, gamma            | `hp / hp_max`, exact              |
| `Route` — timed button holds + frame correlation | any framerate wobble, any enemy nudging you  | real XYZ + heading → pathfinding  |
| `mss` capture at ~30 fps                        | is the hard speed ceiling of the whole loop  | (only a DLL fixes this — stage C) |

And the calibration problem disappears with it: no `flyds1 calibrate`, no
threshold invariants, no "boss_bar_present must be below boss_dead_below".

The reward becomes what the project actually wants to reward, and the
navigation stops being a replayed recording and becomes navigation.

### What this cannot be (read this first)

I cannot verify any of the following from this container, and will not pretend
otherwise:

* **No Windows, no game, no offsets.** I cannot find a single pointer path, and
  I cannot check one you give me. Every address in this project has to come
  from you running a probe tool against *your* build, or from a community
  pointer table for exactly that build.
* **A DLL I cannot test.** `mingw-w64` is not installed here but is available
  as a package, so I *can* cross-compile a `.dll`. I cannot load it, inject it,
  attach a debugger to it, or see it crash. A DLL that crashes the game's
  render thread looks identical, from here, to one that works.
* **Build-specific.** Dark Souls Remastered has had patches; offsets differ per
  version, and PTDE (`DARKSOULS.exe`, 32-bit) and Remastered
  (`DarkSoulsRemastered.exe`, 64-bit) share almost nothing. The plan must
  therefore *discover* addresses at runtime, not hardcode them.
* **Offline only, always.** Reading memory is passive; writing it is
  modification. Do this in offline mode, with a backed-up save. FromSoftware
  bans for modified online play, and that is a rule about their service, not
  about this code. Every stage below assumes single-player, offline, and the
  write-capable stages default to disabled.

This asymmetry — I can write and unit-test *structure*, you must supply and
verify *addresses* — is why the plan is ordered the way it is. Each stage is
useful on its own, and each later stage needs strictly more trust than the one
before.

### Stage A — read-only external memory (no DLL, no compiler)

`OpenProcess` + `ReadProcessMemory` through `ctypes`. Pure Python, no build
step, no injection, nothing loaded into the game. This is where most of the
value is.

New module `src/flyds1/game/memory.py`:

```python
class ProcessMemory:
    """Open a process by executable name; read typed values at an address."""
    def __init__(self, exe_name: str, *, write: bool = False) -> None: ...
    def module_base(self, name: str | None = None) -> int: ...
    def read(self, addr: int, size: int) -> bytes: ...
    def read_struct(self, addr: int, fmt: str) -> tuple: ...     # struct.unpack
    def resolve(self, chain: PointerChain) -> int | None: ...     # None if broken

@dataclass(frozen=True)
class PointerChain:
    """base + offsets[0] -> deref -> + offsets[1] -> deref -> ... -> + last."""
    base: int | str          # absolute, or a module name for module-relative
    offsets: tuple[int, ...]
```

`PointerChain.resolve` must return `None` on a broken link rather than raise:
during a loading screen the game's world pointers are legitimately null, and
that is a normal state to be in, not an error.

New module `src/flyds1/game/scan.py` — AoB signature scanning, so the pointer
*bases* are found in the running process instead of hardcoded:

```python
def find_pattern(mem, pattern: str, *, module: str | None = None) -> list[int]:
    """'48 8B 05 ?? ?? ?? ?? 48 85 C0' -> every match in the module's text."""
def resolve_rip_relative(mem, hit: int, *, instr_len: int, operand_at: int) -> int:
    """x86-64 LEA/MOV rip-relative operand -> the absolute address it names."""
```

Signatures survive patches far better than absolute offsets do, which is the
whole reason to prefer them. They are still build-specific, but they break
much less often.

New module `src/flyds1/game/darksouls.py` — the game-specific layer, and the
*only* place with Dark Souls knowledge in it:

```python
@dataclass
class GameState:
    valid: bool                # False during loading / before the world exists
    player_hp: int; player_hp_max: int
    player_sp: int; player_sp_max: int      # stamina
    position: tuple[float, float, float]
    heading: float                          # radians
    animation_id: int
    area: tuple[int, int]                   # map/block id
    death_count: int
    boss_hp: int | None; boss_hp_max: int | None
    boss_position: tuple[float, float, float] | None

class DarkSoulsReader:
    def __init__(self, layout: MemoryLayout) -> None: ...
    def attach(self) -> bool: ...
    def read(self) -> GameState: ...
```

and `MemoryLayout` as **data, not code** — a YAML file in `configs/memory/`,
one per build, exactly like `NT_SIGNS` is data:

```yaml
# configs/memory/dsr_1.03.1.yaml
exe: DarkSoulsRemastered.exe
bases:
  world_chr_man:
    pattern: "48 8B 05 ?? ?? ?? ?? 48 85 C0 74 ?? 48 63 48"
    rip: {instr_len: 7, operand_at: 3}
fields:
  player_hp:   {chain: [world_chr_man, 0x68, 0x3E8, 0x18C], type: i32}
  position:    {chain: [world_chr_man, 0x68, 0x68, 0x28, 0x10], type: 3f32}
```

The offsets above are **placeholders and certainly wrong**. They are there to
fix the file's shape. Filling them in is a task for stage A.3.

**A.1 — the plumbing (testable here, no game needed).** `ProcessMemory` with
an injectable reader so the whole pointer-chain, struct-decode and
signature-scan logic can be unit-tested against a fake address space: a dict of
`addr -> bytes` standing in for the process. Tests for a broken chain returning
`None`, for rip-relative arithmetic against a known-answer byte sequence, for
a wildcard pattern matching at a known index. This is real work and it is
verifiable without Windows, which is exactly why it comes first.

**A.2 — `flyds1 probe`, a self-validating offset finder.** You run it, it
tells you whether a layout file is right, and it does so by *behaviour*, not by
trust:

```
flyds1 probe --layout configs/memory/dsr_1.03.1.yaml
  attach DarkSoulsRemastered.exe (pid 12345)        ok
  world_chr_man  pattern found at 0x7FF6...         ok
  player_hp      842 / 842      plausible (0 < hp <= hp_max)   ok
  position       (-13.4, 49.2, 88.1)
  >>> walk forward for 3 seconds <<<
  position       (-11.9, 49.2, 82.7)  moved 5.8 units          ok
  animation_id   changed 17 times                              ok
  boss_hp        chain broken (null) -- expected outside a fight
```

Same idea as `flyds1 doctor` and `flyds1 selftest`: never report a green
checkmark for something that was not exercised. A layout that passes `probe`
is trustworthy; one that has not been probed is a guess.

**A.3 — filling in the layout.** Two routes, and you can use both:
1. Community pointer tables. Souls speedrunning and practice tools
   (DSR-Gadget and similar) have published pointer paths for HP, position,
   animation and game speed for DSR builds for years. That is the fast route.
   I cannot fetch or verify them from here; you can, and `probe` will tell you
   whether what you found is right.
2. Cheat Engine yourself: HP is a trivial 4-byte scan (get hit, scan for the
   decrease), then "find what accesses this address" up the chain to a static
   base. Position is three adjacent floats near it.

Either way `probe` is the judge, not me.

**A.4 — wire it into the env.** `MemoryStateSource` alongside the pixel
detectors, and `BossConfig.state_source: "pixels" | "memory" | "auto"`:

* `reward.py` gains a memory path: `Δboss_hp` and `Δplayer_hp` as exact
  integers instead of bar brightness.
* `detectors.py` stays, as fallback and as a cross-check. Worth adding a
  `--compare` mode to `probe`: run both, report disagreements. If the pixel
  detector and the memory reader disagree about death, one of them is wrong and
  you want to know which before a training run, not after.
* `boss.py` gets true episode boundaries: death is a counter increment, a kill
  is `boss_hp == 0`, and neither needs a threshold.

**A.5 — real navigation.** With XYZ and heading, `navigation.py` gains a second
mode next to the recorded `Route`: a waypoint-following controller. Error
between current and target position, rotated into the player's frame, gives
"turn left/right" and "forward"; a waypoint is reached inside a radius. That is
*pathfinding on a recorded polyline* rather than blind replay — it recovers
from being knocked around, which the timed version cannot. Full mesh
pathfinding is not needed and not planned: the bonfire-to-fog-gate path is a
line, and following a line robustly is the entire problem.

This stage is the one I would do next, and I would stop here for a while.

### Stage B — writing memory (still no DLL)

Same `ProcessMemory`, `write=True`, `WriteProcessMemory`. Off by default,
behind `--allow-writes`, and refusing to run unless the layout file declares
itself write-verified.

What this buys, in value order:

1. **Game speed.** The one thing you asked for directly. Dark Souls has a speed
   multiplier in memory (it is what practice tools expose as a speed slider),
   and it is a `float` write, not a hook. At 3× the training loop collects 3×
   the experience per wall-clock hour. Caveats I will not paper over: physics
   and animation scale with it, so an agent trained at 3× may not transfer to
   1× — treat the speed as part of the environment, log it, and validate the
   final policy at 1×. And the capture rate does *not* scale (stage C), so at
   high speeds the fly sees a coarser slice of time.
2. **Instant reset.** Write the bonfire coordinates + full HP instead of dying
   and waiting ~10 s for a reload. An episode reset drops from ~15 s to
   ~0.1 s. For a run that needs thousands of episodes this matters more than
   the speed multiplier does.
3. **Pause / unpause for lockstep.** Write the pause flag, let the brain
   integrate, unpause. Removes the "the brain was too slow this frame"
   nondeterminism from the loop entirely.
4. **Input without the OS.** Write the game's own input state instead of
   `SendInput`. This deletes the focus problem — `_FocusGuardedBackend` exists
   only because `SendInput` goes to whatever window is focused. Writing the
   game's input buffer goes to the game, always, and you can alt-tab freely.
   Needs the input struct, which is a harder find than HP.

A new `MemoryInputBackend(InputBackend)` slots in beside
`PyDirectInputBackend` with no other changes: the backend interface is already
`_press`/`_release`/`move_mouse`.

### Stage C — the actual DLL

After A and B, what is genuinely left that needs code running *inside* the
process? Honestly: three things.

1. **Framebuffer capture without `mss`.** Hook `IDXGISwapChain::Present`, copy
   the back buffer, hand it over a shared-memory ring. This is the only way
   past the ~30 fps capture ceiling, and therefore the only way the speed
   multiplier from stage B actually helps the *vision* rather than just the
   reward. This is the strongest argument for a DLL.
2. **Calling the game's own functions.** Reload an area, respawn enemies,
   trigger a warp properly instead of teleporting a character through a wall
   into a broken collision state. Writes can fake some of this; calling the
   real function is the correct version of it.
3. **True frame lockstep.** Hook the update function, block it while the brain
   thinks, release it. Externally you can only approximate this with a pause
   flag polled at whatever rate you poll it.

Everything else on the "we need a DLL" list turns out to be a write, and stage
B already does writes. That is a finding worth stating plainly: **the DLL is
stage three, not stage one, and most of what you want from it arrives before
it.**

If and when it is built:

```
native/flyds1_hook/
  dllmain.cpp        # DllMain, thread, teardown on unload
  present_hook.cpp   # swapchain Present -> copy back buffer
  shared_frame.cpp   # named shared memory: header + double-buffered BGRA
  ipc.cpp            # named pipe: commands in, state out
  CMakeLists.txt     # MSVC preferred; mingw-w64 cross-build as fallback
```

Python side: `src/flyds1/game/hook.py` — attaches to the shared memory as a
`FrameSource` (so it drops straight into `make_frame_source`, no env changes)
and speaks the pipe protocol for pause/step/reset.

Non-negotiables for anything injected, because the failure mode is a corrupted
save or a hard lock:

* Hook installation on its own thread, never in `DllMain`.
* Clean unhook on unload; the game must survive eject/reinject cycles.
* The DLL never decides anything. No policy, no reward, no game logic — it
  copies frames and forwards commands. All intelligence stays in Python where
  it can be tested.
* A "detached" mode where the hook is loaded but does nothing, to isolate
  crashes caused by presence from crashes caused by activity.
* You test it. I will write it, cross-compile it, and document exactly what to
  check — and I will say in the commit message that it has never been run.

Injection itself: a separate small loader, or one of the existing mod loaders.
Not `flyds1`'s job.

### Ordering, honestly

```
A.1 plumbing + tests   <- I can do this now, fully verifiable here
A.2 flyds1 probe       <- I can write it; you run it
A.3 layout file        <- you (community table or Cheat Engine)
A.4 env integration    <- unblocks exact reward + exact death
A.5 waypoint control   <- unblocks real bonfire->fog-gate navigation
---------------------------------- big value delivered here
B   writes: speed, instant reset, pause, input    <- needs more offsets
---------------------------------- training throughput fixed here
C   DLL: Present hook, lockstep, game calls       <- I cannot test it
```

---

## Part 2 — a real instrument panel

### What is wrong with the current viewer

`live.py` renders four panels server-side into one PNG filmstrip, writes it to
disk, and a browser polls it every 200 ms. Consequences:

* **It shows almost nothing.** One eye (`sorted(front_end.eyes)[-1]` — the
  right one, always), one pooled motion channel, and 24 unlabelled DN bars.
  27 000 neurons are simulated and roughly 1 500 of them are visible.
* **No hemispheres, no cell types, no time.** You cannot see which half of the
  brain is driving, you cannot see T4 vs T5 vs LC4, and every frame erases the
  last one — a panel that only shows *now* cannot show a decay or a buildup.
* **The render cost is on the hot loop.** `facet_image` is cached
  (`_FACET_LAYOUTS`) but still rasterises every facet in Python every frame,
  then PNG-encodes ~960 px of filmstrip, inside the 33 ms step budget.
* **Documented findings are invisible.** `docs/results.md` reports that
  modulation depth collapses 2.0 → 0.010 from photoreceptors to DNs and that
  DNs need 8–16 frames to hear about a stimulus. That is the central finding of
  the project and the live view cannot show it.

### The change: ship state, render in the browser

Send **numbers**, not pixels. The server publishes a small JSON state per
frame; the browser draws it on a canvas. The eye geometry is static, so it is
sent once at startup and never again.

```
flyds1 live
  |
  +-- layout.json   once:  facet coords per eye, cell-type groups, DN labels,
  |                        decoder weight matrix, key names
  +-- state.json    30/s:  ~2-4 KB of floats -- every panel below
  +-- frame.png     ~10/s: the game frame, the one thing that must be a bitmap
  +-- index.html    once:  canvas renderer, ~600 lines, no dependencies
```

This *reduces* hot-loop cost rather than raising it: no rasterising, no PNG
encode except the game frame at a third of the rate. And it makes the viewer
richer and cheaper at the same time, which is the rare case where there is no
trade to weigh.

### Panels

**1. Both eyes, side by side.** `front_end.eyes` already holds a left and a
right retina with per-facet coordinates; the viewer currently throws one away.
Two hex maps, photoreceptor contrast, signed colour. Toggle to motion.

**2. Hemisphere balance — "which half of the brain is she using".** This is
the one you asked for by name, and the data is already there:
`network.neurons.side` is `"left"`/`"right"`/`"center"` for every row, in
matrix-index order, and the synthetic generator fills it in for every columnar,
tangential, projection and descending cell.

```python
left  = rates[neurons.side == "left"].mean()
right = rates[neurons.side == "right"].mean()
balance = (right - left) / (right + left + eps)     # -1 .. +1
```

Rendered as a centre-zero horizontal gauge plus a rolling 10-second trace, and
split by stage (optic lobe vs central vs descending) — because a balance that
exists in the optic lobe and vanishes by the DNs is precisely the project's
known problem, made visible.

**3. Population flow — sensory → optic → projection → central → descending.**
One column per `super_class`, showing mean rate, max rate, active fraction and
modulation depth (rolling std). Read left to right this is the signal
transmission chain from `net/diagnostics.py`, live. When the drive dies before
the DNs you will *watch* it die, at the stage it dies.

**4. Cell-type panels.** `neurons.mask_cell_type` already does substring
matching, so the groups come for free:

* **T4a–d / T5a–d** — eight small hex maps, ON and OFF motion by preferred
  direction. Four directions × two contrast polarities is the fly's entire
  elementary motion detector array, and seeing which subtype lights up tells
  you what direction it thinks the world is moving.
* **LC4 / LPLC2** — the looming detectors. Pooled into a single **threat
  gauge**: in a real fly these drive escape. If the fly ever learns to dodge,
  this needle moves first.
* **HS / VS** — tangential cells, horizontal and vertical optic flow, as two
  signed needles. This is the fly's "am I turning / am I rising" sense.

**5. DN → key wiring.** The interesting panel. 24 DN bars on the left, the
action names on the right, and a line between them whose brightness is
`|decoder_weight| × dn_rate` — the actual per-step contribution, not just the
weight. `DescendingDecoder.contribution()` already returns `|weight|`; this
needs the elementwise product. Pressed keys highlight. You can then see *which
descending neurons pressed which key*, which is the one causal question the
current bar chart refuses to answer.

**6. Time series, rolling ~10 s.** Reward, player HP, boss HP, L/R balance,
threat gauge, DN mean rate. Sparklines, shared time axis, the game frame's
step marked. Without this, a viewer cannot show causality — "the threat gauge
spiked, then it rolled" is a two-panel-in-time statement.

**7. Status.** What `format_stats` prints today, plus: route waypoint and
progress, deaths this session, window focus (is input actually going to the
game?), dry-run banner, steps/s versus target, and whether the state source is
pixels or memory. The focus and dry-run indicators go here because both have
already cost a debugging session each.

### Implementation

```
src/flyds1/viz/
  __init__.py
  groups.py     PopulationIndex: neuron table -> index arrays, built ONCE
  state.py      FlyState dataclass + to_json(); rolling history ring buffer
  assets/
    index.html  canvas renderer + layout, no external dependencies
    fly.js
    fly.css
live.py         becomes a thin publisher: serve assets, write state.json
```

`PopulationIndex` is the part that makes this affordable. Every group — per
side, per super_class, per cell type — is resolved to an index array once at
startup, so each frame is a handful of `rates[idx].mean()` calls over 27 000
floats. That is tens of microseconds, against the tens of milliseconds the
current PNG path spends.

Ordering:

1. `PopulationIndex` + `FlyState` + `to_json`, with tests: groups non-empty,
   sides balanced, JSON round-trips, no NaN ever reaches the browser (a NaN
   silently blanks a canvas panel, which reads as "the fly is dead" — worth a
   test of its own).
2. `layout.json` and the canvas skeleton: eyes, hemispheres, DN wiring. That
   is already more than today's viewer.
3. Population flow and time series.
4. Cell-type panels and the threat gauge.
5. Delete the server-side filmstrip path; keep `plotting.py` (`selftest` and
   the docs use it, and it stays the right tool for a static PNG).

`--viewer classic|panel` during the transition, defaulting to `panel` once the
tests are green, so a broken viewer never blocks a training run.

### What Part 2 does not need

No React, no plotting library, no WebSocket. A canvas, `fetch`, and a
`requestAnimationFrame` loop. The current viewer's best property is that it
cannot steal focus from the game and cannot touch the environment; both survive
this unchanged, and adding a framework would risk the first one for nothing.

---

## Interaction between the two parts

Stage A makes the status panel honest (exact HP, exact death, true position)
and adds a minimap panel — player and boss positions with the route polyline —
which pixels can never provide. Stage C's `Present` hook makes the game-frame
panel free and lets it run at full rate instead of a third.

But they are independent: Part 2 is entirely within this repo, needs no
offsets, no Windows and no compiler, and every piece of it is testable here.
Part 1 stage A.1/A.2 likewise. Everything past A.3 needs you.

If one thing gets built next, it should be **Part 2 steps 1–2 plus Part 1
stage A.1–A.2**: the richest viewer the current data supports, and the probe
tool that tells you whether a memory layout is real. Both are verifiable
without leaving this machine, and both unblock everything after them.
