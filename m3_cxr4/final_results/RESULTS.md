# MedVision-CXR4: results (Kaggle T4, 14 Nov 2026)

**Data:** COVID-19 Radiography Database, 21,165 X-rays. 3,361 images (15.9%) fall into 995 near-duplicate groups (MD5 + dHash, including mirrored copies); each group is kept in one split.
**Split (group-aware, stratified):** about 15.4k train, 2,874 validation, 2,890 test.
**Runs:** 68 trained or ensembled runs, 0 failures (main: 18 runs + 6 ensembles + Grad-CAM; audits: 44 runs). Each model was trained with 3 seeds for 10 epochs, and every number below is on the held-out test set.

## E1 + E2: model comparison (duplicate-aware split, mean ± SD over 3 seeds)
| Model | Family | Accuracy | Macro-F1 | Macro-AUROC | ECE → after temperature scaling |
|---|---|---|---|---|---|
| **Ensemble (all 6, soft voting)** | Ensemble | **0.9594 ± 0.0008** | **0.9664 ± 0.0003** | **0.9959 ± 0.0001** | 0.013 |
| ConvNeXt-Tiny | CNN | 0.9573 ± 0.0016 | 0.9633 ± 0.0017 | 0.9950 ± 0.0002 | 0.031 → 0.010 |
| Ensemble (greedy, chosen on validation) | Ensemble | 0.9571 ± 0.0016 | 0.9634 ± 0.0014 | 0.9956 ± 0.0003 | 0.010 |
| Swin-T | Transformer | 0.9564 ± 0.0009 | 0.9626 ± 0.0013 | 0.9950 ± 0.0002 | 0.025 → 0.009 |
| EfficientNet-B0 | CNN | 0.9537 ± 0.0013 | 0.9614 ± 0.0010 | 0.9946 ± 0.0003 | 0.025 → 0.009 |
| DenseNet-121 | CNN | 0.9539 ± 0.0018 | 0.9601 ± 0.0013 | 0.9945 ± 0.0004 | 0.026 → 0.006 |
| ResNet-50 | CNN | 0.9531 ± 0.0024 | 0.9596 ± 0.0006 | 0.9945 ± 0.0004 | 0.023 → 0.007 |
| ViT-B/16 | Transformer | 0.9502 ± 0.0030 | 0.9578 ± 0.0032 | 0.9936 ± 0.0002 | 0.028 → 0.012 |

- All six backbones land within 0.6 points of macro-F1 of each other. On McNemar tests (seed 0, Holm-corrected) **no pairwise difference is significant**, so the architecture family matters little on this dataset.
- The all-6 ensemble is best on every metric. The ensemble picked greedily on validation does *not* beat it.

**Per class (all-6 ensemble, seed 0)**
| Class | Sensitivity | Specificity | F1 | Test images |
|---|---|---|---|---|
| COVID | 0.980 | 0.998 | 0.986 | 494 |
| Lung Opacity | 0.922 | 0.982 | 0.939 | 855 |
| Normal | 0.970 | 0.951 | 0.958 | 1,366 |
| Viral Pneumonia | 0.994 | 0.998 | 0.983 | 175 |

Most errors are Lung Opacity ↔ Normal: 65 Lung Opacity images were called Normal and 35 Normal images were called Lung Opacity.

**Selective prediction (all-6 ensemble):** answering only the 95% / 90% / 80% most confident cases gives 97.8% / 98.7% / 99.2% accuracy.

## E3: leakage (image-level split vs duplicate-aware split, macro-F1)
| Model | Duplicate-aware | Image-level (leaky) | Inflation |
|---|---|---|---|
| ResNet-50 | 0.9596 | 0.9640 | +0.44 pt |
| EfficientNet-B0 | 0.9614 | 0.9655 | +0.41 pt |
| Swin-T | 0.9626 | 0.9665 | +0.39 pt |

Letting near-duplicates cross splits inflates every model's score a little, every time.

## E5: shortcut audit (macro-F1)
| Model | Full image | Lung crop | Lungs only (outside masked) | Lungs removed |
|---|---|---|---|---|
| ResNet-50 | 0.9596 | 0.9537 | 0.9391 | 0.9454 |
| EfficientNet-B0 | 0.9614 | 0.9585 | 0.9398 | 0.9423 |
| Swin-T | 0.9626 | 0.9533 | 0.9360 | 0.9455 |

**Key finding:** with the lungs blacked out, the models still reach 0.942–0.946 macro-F1, slightly *higher* than when they can see only the lungs (0.936–0.940). A large part of the signal therefore lies outside the lungs (source hospital, scanner, text markers, borders). This is the dataset-bias (shortcut) problem, and the main caveat for clinical use.

## E4: Grad-CAM lung-focus audit (400 test images, seed 0)
| Model | Share of heatmap inside lungs | Pointing game (peak inside lungs) |
|---|---|---|
| EfficientNet-B0 | 0.371 | 56.5% |
| Swin-T | 0.354 | 52.3% |
| ResNet-50 | 0.328 | 39.8% |
| DenseNet-121 | 0.307 | 44.3% |
| ConvNeXt-Tiny | 0.274 | 25.8% |
| ViT-B/16 | 0.195 | 14.0% |

The lungs cover 23.9% of the image on average. EfficientNet-B0 and Swin-T concentrate on the lungs most; ViT-B/16 attends to the lungs less than chance. Accuracy and lung focus do not go together: ConvNeXt is the most accurate single model but ranks second-to-last on lung focus.

## E6: calibration
Raw ECE is 0.023–0.031 for single models. Temperature scaling, fitted on validation (T ≈ 1.4–1.7), reduces it to 0.006–0.012. All models were over-confident before scaling.

## E7: ablations (EfficientNet-B0, macro-F1, 2 seeds)
| Setting | Macro-F1 | Change |
|---|---|---|
| Full pipeline (3 seeds) | 0.9614 | — |
| No CLAHE | 0.9622 | +0.08 |
| No augmentation | 0.9627 | +0.13 |
| No class weights | 0.9624 | +0.10 |
| **No ImageNet pretraining** | **0.9083** | **−5.31** |

Only ImageNet pretraining matters; the other choices are within seed noise. Seed 2 of the ablations was not run because of the Kaggle 12-hour limit.

## Files
`table_main.md/.csv`: every configuration, mean ± SD · `table_ci.csv`: bootstrap 95% CIs · `pairwise_mcnemar.csv`: significance tests · `all_runs.csv`: every run · `cam_audit_summary.csv` · figures `fig_*.png`, `gradcam_gallery.png` · `logs/`: the Kaggle run logs.
