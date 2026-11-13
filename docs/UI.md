# The Simstack window

One window: a menu bar on top, the 3D view in the middle, docked panels
(Controls on the left; Plots, Inspector and Performance as tabs below the view),
and the transport bar along the bottom. Every panel can be closed, re-opened
from **View**, dragged to another dock or floated; **View → Reset layout**
puts them back.

## Transport bar (bottom)

| Control | What it does |
| --- | --- |
| ↻ | Restart the run. Settings you changed are kept (use **Defaults** in Controls to drop them). Shortcut `R`. |
| ⏮ | Step back to the previous keyframe (`,`). |
| ▶ / ⏸ | Play / pause (`Space`). |
| ⏭ | Step forward: one chemistry step live, one keyframe in playback (`.`). |
| speed | Simulated time per wall second. NFT presets run from real time to one week per second; chemistry from 0.01× to 100×. **max** spends the whole frame budget. If the machine can't keep up, simulated time runs slower and the frame rate stays smooth. |
| time | Simulated time (NFT: day and clock time). |
| **Bake…** | Simulate ahead as fast as the machine allows, recording keyframes, then play the recording back (below). |
| timeline | Recorded range (blue), keyframe ticks, playhead. Drag to scrub. While baking, the bake's progress shows in orange. |
| LIVE / PLAYBACK / BAKING | State badge. In playback, click it to jump to the live end. |
| right | Speed actually achieved and frames per second. |

## Timeline, playback and baking

While the scene runs it records **keyframes**: complete copies of the scene's
state (every scene keeps all of its state in one arena, so a keyframe is one
copy and restoring it is one copy back). They go into a 16 GB address range
reserved at start-up and committed by the kernel only as it fills; nothing is
allocated while running. When it is full, every other keyframe is dropped and
the spacing doubles, so the whole run stays scrubbable.

- **Scrub** the timeline to see any recorded moment: plots, the 3D view and the
  Inspector all show the state at that keyframe.
- **Play** in playback moves through the recording at the chosen speed without
  simulating. At the end of the recording the scene goes live again.
- **Edit anything** (a parameter, a preset) while in playback and the run
  **forks** there: the recorded future is dropped and the scene simulates on
  from that keyframe with the new settings.
- **Bake…** asks for a duration and a keyframe count, then runs as fast as
  possible and plays the result back from its start. For the DBD reactor the
  bake can switch to **reference precision** (f64 lanes, rtol 1e-5, 1 ms
  coupling) first, so a slow high-accuracy run can be inspected at any speed
  afterwards. The popup shows the memory a bake will take.
- `--bake T` bakes at start-up (`--bake 30d` for an NFT month, `--bake 2` for
  two seconds of chemistry).

The simulation is deterministic: restoring a keyframe and running on gives
bit-identical results. `--timeline-check` (headless) checks this: it records a
keyframe at the half-way frame, finishes, restores it, runs the second half
again and compares the state hashes.

```sh
./build/simstack --scene dbd --headless 600 --timeline-check
./build/simstack --scene nft --preset paw-skid --headless 300 --timeline-check
```

## 3D view

| Mouse / key | |
| --- | --- |
| Left drag | Orbit |
| Right or middle drag, `Shift` + left drag | Pan |
| Wheel | Zoom toward the point under the cursor |
| Hover | Tooltip with the key values of the point under the cursor |
| Click | Select it: the Inspector shows everything about it, with a profile along its channel or tube |
| Double-click | Focus the camera on the selection |
| `F` | Frame the whole scene |
| `Home` | Reset the camera |
| `F1` | Auto-orbit |
| `Esc` | Clear the selection, close menus |

What the Inspector shows:

- **NFT channel water**: film temperature, rise above the inlet, dissolved
  oxygen, EC and pH, every ion, film depth, root fraction, velocity, channel
  flow; a profile along the channel of the layer shown in the view.
- **NFT plant**: dry mass and its share of the crop's maximum, age, growth
  limits (temperature, nitrogen, oxygen), water use, N, K and Ca uptake, root
  oxygen demand, harvests, bolting heat dose.
- **DBD zone**: gas temperature, pressure, humidity, discharge power and
  frequency, every species above 1e-9 ppm, solver steps and rejects; the
  species' profile along the tube with the zone marked.
- **Sweep point**: its power and humidity, and the same chemistry.

**View** toggles the overlays: floor grid, labels, colour legend, axes gizmo,
mouse hints, and in the NFT scene the channel water and the plants separately.
The colour legend and layer picker sit in the top-left of the view.

## Controls

**Simple** shows the parameters you usually change; **Advanced** adds the
solver, zone count, chemistry step, electron-impact yields (DBD), the PVC
hydraulics and the water and root details (NFT). **Defaults** restores every
setting of the scene (NFT: reloads the selected preset) and restarts.

## Menus

- **File**: export the CSV (path editable in the menu), save a screenshot
  (`simstack_shot_NNN.ppm` in the working directory), quit (`Ctrl+Q`).
- **Scene**: switch scenes; NFT presets; one module of the PAW skid in the DBD
  reactor scene.
- **Simulation**: play/pause, step, restart, max speed.
- **View**: panels, camera, overlays, UI scale (automatic from the window
  height: 125 % from 1400 px, 150 % from 2000 px), reset layout.
- **Help**: shortcuts (`F2`), about.

## Scripted screenshots

`--frames N --screenshot out.ppm` quits after N frames and writes the last one.
`--pick FX,FY` hovers and selects the point at (FX, FY), as fractions of the 3D
view, so the tooltip and the Inspector can be captured too:

```sh
./build/simstack --scene nft --frames 200 --pick 0.55,0.35 --screenshot nft.ppm
```
