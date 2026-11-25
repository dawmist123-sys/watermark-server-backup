# HVS-GM experiment reproduction

This package implements the revised manuscript's experiments for independent reruns. Commands produce fresh outputs with their own scores and timing. Use the same manifest, checkpoint and preprocessing when comparing runs.

## Experiment map

| Manuscript | Entry point | Output |
| --- | --- | --- |
| §3, Eqs. (4)–(17) | `pipeline.py`, `core.py`, `protocol.json` | HVS-GM embedding, empty-prompt DDIM inversion, Pearson magnitude score |
| §4.2, Tables 4 and 5 | `benchmark.py`, `baselines.py`, `stable_signature.py` | Paired scores, ROC AUC and empirical TPR at FPR ≤1%; rerun on SD v1.5, v2.0 and v2.1 using the corresponding checkpoint |
| §4.2, additional attacks | `benchmark.py`, `purification.py` | Blur, noise, brightness, crop locations, 256-pixel resize control and DiffPure at t=50 and t=100 |
| §4.1.2, Table 6 | `prepare_coco.py`, `run_table6.py`, `metrics.py` | Independent threshold, realized TPR/FPR, and paired bootstrap ROC intervals |
| §§4.3–4.4, Tables 7 and 8 | `quality.py`, `benchmark.py`, `diagnostics.py` | FID, CLIP, paired LPIPS, parameter sensitivity and texture proxy measurements |
| §4.5, Table 9 | `benchmark.py`, `diagnostics.py` | Component removal, complex-spectrum control, energy-matched comparison, and inversion diagnostics |
| §4.6, Table 10 | `benchmark.py`, `baselines.py`, `stable_signature.py` | Per-image generation and detection timing, summarized as mean and sample standard deviation |

The HVS-GM defaults are 50 DDIM updates, 10 preliminary denoising updates, guidance 7.5 for generation and 1 for inversion, a 15×15 reflective window, 95th-percentile normalization, sigmoid coefficient 5, bounds 1.6–2.2, Gaussian widths 0.50 and 0.70, four channel-specific templates and a radius-10 band. The rerun uses CPU PyTorch draws for initial/key latents, checkpoint scheduler settings, DDIM eta=0, 512×512 RGB images and the VAE posterior mode. Crop size uses retained area and bicubic resizing. Rotation uses bicubic interpolation, a fixed canvas and black fill.

## Environment and input

Use local Diffusers-format checkpoints. For RTX PRO 6000, use a Blackwell-compatible PyTorch build. Install model dependencies and, for perceptual metrics and DiffPure, the extra dependencies below.

```sh
python -m pip install -r requirements.txt
python -m pip install -r requirements-quality.txt
python -m unittest -v test_core

python prepare_coco.py --train captions_train2017.json --validation captions_val2017.json \
  --exclude-test-ids historical_test_ids.json --output manifest.csv
```

The manifest records the COCO ID, caption, split and seed. For Table 6 use 1,000 training IDs for clean calibration negatives and 1,000 validation IDs for matched test pairs. Sampling and bootstrap use seed 20262026. Supply the known historical test IDs as a JSON array. Reuse the manifest across methods to compare the same prompts and pair outputs by ID.

## Generation, attacks and thresholds

```sh
C=/path/to/sd-v1.5
M=manifest.csv
python pipeline.py --checkpoint "$C" generate --prompt "a view of the coast" --seed 200000 --output result.png
python pipeline.py --checkpoint "$C" detect result.png --score pearson

python benchmark.py --checkpoint "$C" --manifest "$M" --score negative_l1 --output hvs_table4 --save-images
python benchmark.py --checkpoint "$C" --manifest "$M" --score negative_l1 --output hvs_table5 \
  --conditions clean jpeg50 crop50 rotation75 crop50-random crop50-offcenter crop50-asymmetric \
  crop50-upper_left crop50-upper_right crop50-lower_left crop50-lower_right crop75 crop75-random \
  crop75-offcenter crop75-asymmetric crop75-upper_left crop75-upper_right crop75-lower_left \
  crop75-lower_right blur2 noise005 brightness15

python run_table6.py --checkpoint "$C" --manifest "$M" --output table6_run
python metrics.py table6_run/calibration.csv table6_run/test.csv
```

Table 4's historical HVS-GM ROC evaluation and the Table 6 evaluation use negative mean absolute Fourier magnitude difference. Section 3.7 defines Pearson correlation on concatenated magnitudes. For Table 6, use the clean-negative 99th percentile with NumPy's `higher` rule and detect with a strict score-above-threshold comparison. Both classes receive the same edit and the fixed threshold is reused across conditions. The ROC work point uses the maximum empirical TPR at FPR ≤1% without interpolation. A small integration check uses `run_table6.py --smoke-per-split 2` and `metrics.py --smoke`.

## Baselines and DiffPure

`baselines.py` uses source and weights from the original implementation layout under `--project-root`. The archive layout uses `tree-ring-watermark-main` and `revision_20261111/baselines`, with compatible dependencies in the active environment and the GaussMarker GNR weights and key under `revision_20261111`. Use the associated author source, key and weights for each method. For a different layout, pass the relevant checkout or nested Python source folder using `--author-source`. For GaussMarker, use `--gnr-checkpoint`, `--gnr-key`, and `--gnr-classifier` to select its matching artifacts. The local archive's PRC environment can be supplied via `PYTHONPATH="$R/revision_20261111/isolated_prc"` in a compatible Python 3.12 environment.

```sh
R=/path/to/watermark_project
python baselines.py --algorithm treering --model "$C" --project-root "$R" --manifest "$M" --out tree_ring_run
python baselines.py --algorithm gaussmarker --model "$C" --project-root "$R" --manifest "$M" \
  --out gaussmarker_run --attacks clean jpeg50 crop50 rotation75

python stable_signature.py --checkpoint /path/to/matching/base-diffusers-model \
  --marked-vae /path/to/sd2_decoder.pth --extractor /path/to/dec_48b_whit.torchscript.pt \
  --key "$STABLE_SIGNATURE_KEY" --manifest "$M" --output stable_signature_run

python purification.py --checkpoint "$C" --manifest "$M" --score negative_l1 \
  --diffpure-source /path/to/DiffPure --diffpure-weights /path/to/guided_diffusion --output diffpure_run
```

Other `--algorithm` values are `prc`, `gaussianshading`, and `grow`. Stable Signature accepts a raw [official decoder](https://github.com/facebookresearch/stable_signature) checkpoint or a converted Diffusers VAE directory. Provide the extractor and binary key matched to that decoder. DiffPure defaults to 100 pairs and measures LPIPS relative to the unedited source image for the resize control and both purification strengths.

## Image quality, sensitivity and ablations

```sh
python quality.py --manifest "$M" --images hvs_table4/images --clip-model ViT-L-14 \
  --clip-weights /path/to/open_clip_model.safetensors --lpips-network vgg --output hvs_quality.json

python benchmark.py --checkpoint "$C" --manifest "$M" --score negative_l1 --output sigma_035 \
  --conditions clean crop50 rotation75 --sigma-mask 0.35 --save-images
python benchmark.py --checkpoint "$C" --manifest "$M" --score negative_l1 --output strong_range \
  --conditions clean crop50 rotation75 --strength-range 2.0 2.6 --save-images

python benchmark.py --checkpoint "$C" --manifest "$M" --score negative_l1 --output equal_energy \
  --conditions clean crop50 rotation75 --variant fixed_equal_energy
python benchmark.py --checkpoint "$C" --manifest "$M" --score negative_complex_l1 --output complex_match \
  --conditions clean crop50 rotation75

python diagnostics.py --checkpoint "$C" --manifest "$M" texture --steps 10
python diagnostics.py --checkpoint "$C" --manifest "$M" inversion --score negative_l1
```

Repeat sensitivity runs using fixed-mask widths 0.35, 0.50 and 0.70; sigmoid coefficients 2.5, 5 and 10 via `--sigmoid-coefficient`; or the bounds compared in Table 8. Score each output with the same CLIP weights and LPIPS backbone. FID uses Inception pool3 features and compares the marked distribution with clean samples. LPIPS pairs marked and clean frames by image ID. The benchmark also accepts `--sigma-window` and the variants `no_fixed_mask`, `no_window`, `no_mcr`, `no_mcre` and `fixed_1_75`. The fixed-mask ablation removes the embedding-side mask. The equal-energy variant matches each image's perturbation norm after modulation and before latent addition. The texture diagnostic compares each requested early update with final-image texture using spatial Spearman correlation and the overlap of top-quartile pixels. The inversion diagnostic compares the direct latent, true-prompt inversion and empty-prompt inversion and reports the corresponding RMSE where applicable.

Use the `clean` condition to obtain comparable timing. HVS-GM generation timing covers the preliminary denoising, perturbation construction and injection, and final denoising. For qualitative comparisons, use the same IDs in the saved result images. For small GPU checks, pass `--limit` to the applicable benchmark. Interpret estimates on such subsets as integration checks.
