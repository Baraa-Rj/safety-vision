# Validation runs

Generated with `ultralytics` 8.4.33 on CPU. These are the only formal accuracy
artifacts in the repo. Re-create with the commands below.

> **LEAKAGE NOTICE (2026-07-12).** Source-frame comparison (filenames stripped
> of Roboflow hashes, pixel-verified identical) showed that **all 33 valid and
> all 18 test images of `data/ppe_dataset/` also appear in the 169-image
> fine-tune training set** (and 22/13 of them in the fall train split). Since
> `models/best.pt` is the fine-tuned model, the `ppe_val_640` / `ppe_test_640`
> numbers below are **train-set performance, not held-out accuracy**. The
> `fall4_*` runs are also mildly affected (35/211 valid and 16/106 test frames
> leak). The authoritative held-out numbers are **`detect/clean_heldout_640`**.

## detect/clean_heldout_640 — `models/best.pt` (AUTHORITATIVE held-out eval)
Fall-dataset valid+test frames minus every frame whose source key appears in
any local training pool (fall train, local ppe train, fine-tune train) and
minus all `training*`-session frames (those sessions fed the published PPE
train split, which cannot be checked locally). 266 images, 760 instances,
deduped by source frame. imgsz=640.

| class  | instances | P     | R     | mAP50 | mAP50-95 |
|--------|-----------|-------|-------|-------|----------|
| fallen | 155       | 0.963 | 0.929 | 0.980 | 0.901 |
| helmet | 136       | 0.962 | 0.941 | 0.977 | 0.899 |
| person | 241       | 0.994 | 0.983 | 0.995 | 0.978 |
| vest   | 228       | 0.985 | 0.952 | 0.992 | 0.953 |
| **all**| **760**   | 0.976 | 0.951 | **0.986** | 0.933 |

Remaining caveat: clean frames still come from the same recorded sessions as
fall-train footage (frame-level split of shared videos) — on-site validation
is still pending.

## detect/ppe_val_640, detect/ppe_test_640 — `models/best.pt` (V1)
PPE detection on the only labelled detection set in the repo
(`data/ppe_dataset/`, re-indexed to `best.pt`'s 4-class namespace —
see `_dataset/ppe4_reindexed_data.yaml`). imgsz=640.

```
yolo detect val model=models/best.pt data=runs/_dataset/ppe4_reindexed_data.yaml split=val  imgsz=640 plots=True
yolo detect val model=models/best.pt data=runs/_dataset/ppe4_reindexed_data.yaml split=test imgsz=640 plots=True
```

| split | images | all mAP50 | all mAP50-95 | all P | all R |
|-------|--------|-----------|--------------|-------|-------|
| val   | 33     | 0.99305   | 0.96294      | 0.98990 | 0.98402 |
| test  | 18     | 0.99066   | 0.96235      | 0.99436 | 0.97949 |

Per-class P/R/mAP for helmet, person, vest are in each run's curves and
confusion matrix. **The `fallen` class has ZERO ground truth in this set, so
fallen-class detection recall and FPR are NOT measured by these runs.** For
fallen-class metrics see `detect/fall4_val_640` / `detect/fall4_test_640` below.

## detect/fall4_val_640, detect/fall4_test_640 — `models/best.pt` (fallen class)
Fall detection on the held-out splits of the 1,056-image 4-class fall dataset
(`falling.yolo26(1).zip` at the repo root — Roboflow `chestx-ibk9e/falling-s0blr`,
CC BY 4.0; same class namespace as `best.pt`: fallen/helmet/person/vest;
739/211/106 train/valid/test). Extract the zip, point a data yaml at it
(nc=4, names as above), then:

```
yolo detect val model=models/best.pt data=<fall4_data.yaml> split=val  imgsz=640 plots=True
yolo detect val model=models/best.pt data=<fall4_data.yaml> split=test imgsz=640 plots=True
```

Fallen-class results (imgsz=640, CPU):

| split | images w/ fallen | fallen inst. | P | R | mAP50 | mAP50-95 |
|-------|------------------|--------------|-------|-------|-------|----------|
| val   | 87               | 107          | 0.943 | 0.929 | 0.975 | 0.898 |
| test  | 42               | 48           | 0.979 | 0.993 | 0.994 | 0.912 |

Caveats: (1) valid/test frames come from the same recording sessions as
training frames (Roboflow frame-level split of shared videos), so these are
in-distribution numbers — on-site footage validation is still pending;
(2) the dataset mixes polygon and box labels; Ultralytics used boxes only.

## classify/fall_cls_val_224 — `models/fall_cls.pt` (V4, proxy only)
Fall **classifier** (NOT loaded by the pipeline) on `data/crops/val`. imgsz=224.

```
yolo classify val model=models/fall_cls.pt data=data/crops imgsz=224 plots=True
```

top1=0.98738. Confusion (fallen positive): TP=101, FN=4, FP=0, TN=212 →
sensitivity=0.96190, FPR=0.00000 over 317 crops. This validates the unused
classifier at 224px crops, not the deployed `best.pt` fall-detection path, and
not at 1080p; it does not substantiate SR2.4 for the deployed system.
