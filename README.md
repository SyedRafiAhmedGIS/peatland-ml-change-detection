# 🌿 Peatland ML & CORINE Change Detection Pipeline · v10.1

End-to-end, **Colab-ready** pipeline that maps peatland land cover from **Sentinel-2** imagery and compares two mapping approaches on identical data: a pixel-wise **Random Forest** and a context-window **U-Net** (segmentation_models.pytorch), with **MC-dropout uncertainty**, paired **block-bootstrap confidence intervals**, and an honest built-in **QC gate**.

Designed to run in a free Colab session: lazy/chunked loading, time budgets per block, skip-don't-simulate policy, and every diagnostic exported to `reports/`.

---

## What it does

1. **Study area & labels** — loads a labelled land-cover raster (CORINE, or the bundled ESA WorldCover proxy) and a study-area bbox over the Irish Midlands bogs.
2. **Sentinel-2 composite** — queries Planetary Computer, picks the clearest scenes per MGRS tile, cloud/shadow-masks via SCL, removes the **per-scene +1000 DN processing-baseline offset** (keyed by solar day, so same-day merged scenes are handled correctly), and builds a median composite with clear-observation diagnostics.
3. **Fair model comparison** — spatially block-split tiles (train/val/test with 2-tile buffers, no leakage), √-balanced class priors used consistently for RF sampling and U-Net loss weights.
4. **Evaluation** — 7-class confusion matrices, per-class F1, 100 m aggregation, and **paired block-bootstrap 95% CIs** (same block draws for both models) on the metrics that matter, including peat-class F1/recall.
5. **Uncertainty (Block 11b)** — MC dropout (T=20) on a fine-tuned copy of the U-Net: predictive-entropy maps, reliability diagrams/ECE, and a rejection curve.
6. **Change detection (Block 12)** — 2012-vs-2018 CORINE transition matrix, automatically skipped with an explanation if proxy labels are in use.

## Results from the included demo run (`reports/`)

Irish Midlands bogs · 8 Sentinel-2 scenes (May–Jul 2018) · 20 test blocks · 1.31 M test pixels · ~3 min total on a Colab GPU.

| Metric (test) | Random Forest | U-Net | U-Net − RF (95% CI) |
|---|---|---|---|
| Accuracy | 0.843 | 0.851 | +0.008 (point diff; no CI exported) |
| Kappa | 0.580 | 0.603 | [−0.017, +0.061] — ns |
| Macro F1 | 0.582 | 0.562 | [−0.038, +0.066] — ns |
| **Peat F1** | 0.330 | **0.389** | [−0.012, +0.141] — suggestive, ns |
| Peat recall | 0.512 | **0.695** | — |

- No statistically significant RF/U-Net difference at 20 test blocks; U-Net maps are visibly more spatially coherent and recover ~18 pts more peat.
- MC dropout: ECE 0.047 (all classes) / 0.058 (peat 1-vs-rest); entropy of wrong pixels 0.80 vs 0.50 for correct; rejecting the most uncertain 30% of pixels buys **+7.9 pts accuracy**.
- QC gate: 16 pass / 3 warn (all three warns: the *Other wetlands* class has 0 pixels in this proxy-label area).

## Repository layout

```
├── peatland_ml_change_detection_v10_1_fixed.ipynb   # the pipeline (run top-to-bottom)
├── reports/                                         # outputs of the included demo run
│   ├── run_info.json            # timings, scene list, versions, config echo
│   ├── data_quality.json        # QC gate results (pass/warn per check)
│   ├── metrics.json             # both models + bootstrap CIs + 100 m aggregation
│   ├── metrics_RandomForest.json / metrics_UNet.json
│   ├── uncertainty.json         # MC dropout settings, ECE, rejection curve
│   ├── spatial_split_tiles.npy  # exact tile-level train/val/test assignment
│   ├── 01_study_area.png        # composite vs reference labels
│   ├── 02_spatial_split.png     # block split with buffers
│   ├── 03_unet_training.png     # loss curves
│   ├── 04_results.png           # confusion matrices + per-class F1
│   ├── 05_test_maps.png         # reference / RF / U-Net maps + peat error map
│   └── uncertainty.png          # entropy map, prediction, reliability diagram
```

## Requirements

Tested with Python 3.13 · torch 2.11 (CUDA) · segmentation-models-pytorch 0.5.0 · scikit-learn 1.6.1 · numpy 2.1.3 · xarray 2025.12 · odc-stac · pystac-client 0.9.0 · rasterio 1.5.2. No `requirements.txt` is pinned on purpose — the first notebook cell prints the exact versions used by any run into `reports/run_info.json`.

## How to run

1. Open the notebook in **Colab** (Runtime → GPU) and run all cells.
2. When prompted, upload your land-cover rasters for each year (CORINE 100 m rasters, or leave the demo proxy in place).
3. Let it run — each block prints its own QC banner; all artifacts land in `reports/`.
4. Zip/share `reports/` after your run — every number in it is traceable via `run_info.json`.

## Known limitations (kept deliberately honest)

- The bundled labels are an **ESA WorldCover 2020 proxy** (herbaceous wetland substituted for peat) — peat-class numbers are proxy numbers, and the 2012-vs-2018 **change detection is skipped** until real CORINE 2012 + 2018 rasters are provided.
- Peat precision is low for both models (~0.24–0.27): predicted-peat is concentrated along bog margins; part of this is proxy-label coarseness.
- Water and Artificial have tiny test support — their aggregated 100 m scores are not meaningful.
- Dark peat is over-flagged as cloud shadow (4.1% vs 0.5% elsewhere); composite gaps remove 1.7% of peat pixels.

## Version notes

- **v10.1** — Block 5 fix: `groupby="solar_day"` merges same-day scenes, so per-scene offsets are now **keyed by solar day** (UTC + lon/15 h) instead of positional; length mismatch with the merged time axis is impossible. Same-day conflicting baselines warn; nearest-day fallback warns; the day→offset map is saved into the composite cache metadata.
- **v10** — upload year validation; per-scene baseline offsets; real nodata-consistency check; slim validation on mid-epoch budget hit; MC-pass budget guard; split + package versions exported.
- **v9** — `matplotlib.patches` import crash fix; raw-label cloud-bias diagnostic; MC-vs-Block-10 comparison; upload wait fix; RF shadowing fix.
