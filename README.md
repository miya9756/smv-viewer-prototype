# SMV Viewer Prototype

A minimal in-browser player for the compressed dynamic 3D-Gaussian scenes of
*SmoothMotionVectors* (SIGGRAPH 2026). The browser downloads a scene's `.zip`, decodes its HEVC
motion streams with ffmpeg.wasm and rasterizes the gaussians with WebGL2. No install, no server
compute, no build step.

It holds every run of the results table except NeRF-DS:

| Dataset | Runs | Scenes |
| --- | --- | --- |
| Neural 3D Video | Ours+Grid4D-H-30-5, Ours+Grid4D-H-60-10, Ours+SpDeF-H-30-5, Ours+SpDeF-H-60-10, Ours+SpDeF-L-30-5, Ours+SpDeF-L-60-10 | coffee_martini, cook_spinach, cut_roasted_beef, flame_salmon_1, flame_steak, sear_steak |
| MeetRoom | high, low | discussion, trimming, vrheadset |

42 scenes, 483 MB. Pick a run and a scene, press **Load**, then **Play**. Fly with drag-to-look
and W A S D / E Q, or switch to orbit (`?nav=orbit` opens in it); double-click the view to reset it.

## Run it locally

ES modules and WebGL2 need `http://`, not `file://`. Serve the folder with any static server:

```bash
python3 -m http.server 8000
```

and open <http://localhost:8000>.

## Layout

| Path | What it is |
| --- | --- |
| `index.html` | the page: picker, WebGL2 splat renderer, GOP streaming window, playback, navigation |
| `decode.js`, `decode_motion.js`, `dequant_worker.js` | the decode path, verbatim from `smoothmotionvector-assets/web/player_browser/` |
| `sw.js` | service worker that keeps the 31 MB ffmpeg.wasm core cached across visits |
| `vendor/` | ffmpeg.wasm and fflate, vendored |
| `scenes.json` | the scene list, with each scene's held-out camera |
| `scenes/<dataset>/<run>/run01/<scene>.zip` | the compressed scenes |

The decode modules are copies. If the dequantization changes, change it upstream and copy the files
again, or the browser will disagree with the reference decoder.

**The GOP layout comes from the run's folder name.** `…-30-5` means step 30 / overlap 5 and
`…-60-10` means 60 / 10. MeetRoom states neither and is 30 / 5. Inside an overlap the two GOPs are
rendered whole and cross-faded.

**To add a scene**, put its zip under `scenes/` with the same layout, and add an entry to
`scenes.json` (`name`, `url`, `mb`, and `cam` with `fovx`, `fovy`, `width`, `height`, `campos`,
`camquat`).

## Licence

CC BY-NC-ND 4.0, except for the third-party material listed in [LICENSE](LICENSE).
