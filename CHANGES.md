# Changes — 2026-06 session

This repo was `git init`'d at `master` (commit `0a382ea` = baseline: SpotEdit + the new
`Qwen_image_edit_plus/` and `FLUX2/` backbone modules + eval scripts). Two **independent**
feature branches were cut off `master` — they touch disjoint files, so they can merge separately.

---

## Branch `flux2-velocity` (4 commits) — velocity reuse for FLUX.2 + fair speedup methodology

### Production code (`FLUX2/`, +38/-6) — **changes the default** to velocity
| file | change |
|---|---|
| `flux2_spot_ultis.py` | add `reuse_mode` to `SpotEditConfig` (default `"velocity"`, `"overwrite"` = old behavior) |
| `flux2_spotedit.py` (dev) | reused tokens flow to source `v=(x_t-x0_orig)/σ` in-loop; old hard-paste + boundary-smoothing now **gated on `reuse_mode=="overwrite"`** (was unconditional → seam-prone) |
| `flux2klein_spotedit.py` | respect `reuse_mode` + add `overwrite` end-block + expose reuse mask via `aux` dict |

### Eval scripts
`flux2_tasks.py` (4 edit types), `flux2_dilation.py` (dilation sweep), `klein_guidance_check.py`,
`qwen_fullbase.py`, `qwen_fair.py`.

### Findings (klein-4B, the GPU-runnable FLUX.2)
- velocity is **clean on localized edits** (no seam); reuse fraction **adapts to edit locality**
  (party hat 90%/1.94×, scarf 70%/1.85×); **global edits** (recolor-all, bg-swap) → 0% reuse →
  graceful **full-compute fallback** (Δ=0, ~1×). So SpotEdit accelerates *localized* edits only.
- dilation **0–1 is the sweet spot** for localized edits — velocity makes the tight border seamless
  (dil0 2.07×/89% reuse … dil4 1.68×/29%, all clean). dilation can't help 0%-reuse cases (→ threshold).
- **Fair speedup** = base & spot at **identical steps, NO CFG, only reuse toggled**:
  Qwen-2509 **2.23×** (`qwen_fair.py`: same loop, 40 steps, reuse off 19.8s vs on 8.9s, Δ2.9/255),
  klein 1.85–2.07×. (Pitfall recorded: comparing single-forward spot vs a CFG 2×-forward base is
  unfair — `qwen_fullbase.py` shows it inflates to 4.27×; do NOT report that.)

---

## Branch `lpips-speed-study` (8 commits) — VAE-LPIPS judge profiling + the T=1 fix

### Production code (`Qwen_image_edit_plus/`, +15/-4) — **opt-in, default behavior unchanged**
| file | change |
|---|---|
| `QwenTokenLPIPS.py` | add `temporal_fix` flag (default `False`); when set, `_safe_unpack` adds the missing singleton time axis |
| `qwen_spot_ultis.py` | `Spotselect` gains method `'LPIPS_fixed'` (uses `temporal_fix=True`) |

### Eval scripts + doc
`lpips_speed.py`, `lpips_speed_fix.py`, `lpips_verify_real.py`, `methods_compare.py`,
`why_diff.py`, `recalib_fixed.py`, `inspect_vae.py`, `decode_t1_vs_t16.py`, `fixed_heatmap.py`,
and **`LPIPS_SPEED_STUDY.md`** (full writeup).

### Findings
- The VAE-LPIPS reuse judge = **138.7 ms ≈ a full VAE decode**; cosine/L4 = ~0.05 ms (2300–3700× faster).
- Root cause: `_safe_unpack` omits the temporal axis → the 5D mean/std broadcast inflates the latent to
  `[1,16,16,128,128]`, so the **3D causal video VAE decodes a phantom 16-frame volume** (~16× compute),
  then averages it away. (Confirmed: `AutoencoderKLQwenImage`/`QwenImageDecoder3d`, `QwenImageCausalConv3d`;
  `decode_t1_vs_t16.py` shows T=1→the dog, T=16→61 colour-corrupted frames.)
- **T=1 fix** = 138→24 ms (**5.8×**) AND better localization (Spearman vs true edit 0.665 > 0.573).
  Score scale shifts non-linearly → recalibrate `threshold` by **target reuse fraction**, not a constant.
- End-to-end (`methods_compare.py`): all judges give **identical output quality**; at default cadence the
  judge is ~8% of runtime; at `select_every_step` LPIPS collapses 2.3→1.51×, the fix recovers it to 1.99×.
  For this edit **cosine/L4 (free) are as good as LPIPS** → candidate default.

---

## Merge status & recommendations
- **Nothing merged to `master` yet.** Working tree on `master` = baseline (velocity/temporal_fix NOT active).
- `flux2-velocity` → **ready to merge** (velocity is a clean default improvement; dev path no longer seams).
- `lpips-speed-study` → keep as a study branch; `temporal_fix`/`LPIPS_fixed` are **opt-in** (need `threshold`
  recalibration before becoming default — see LPIPS_SPEED_STUDY.md §4b/5d).
- Open items: port `target_reuse_fraction` auto-calibration into `SpotEditConfig`; validate cosine-vs-LPIPS
  on a diverse edit benchmark; the real speedup ceiling is the cached-attention step (full-length K/V), not the judge.

## Outputs
All montages/logs are under `eval/results/` and `eval/logs/` (**git-ignored**, regenerable by the scripts above).
