# Validation runs

Generated with `ultralytics` 8.4.33 on CPU. These are the only formal accuracy
artifacts in the repo. Re-create with the commands below.

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
fallen-class detection recall and FPR are NOT measured by these runs.** No
fallen-labelled detection validation set exists in this repo.

## classify/fall_cls_val_224 — `models/fall_cls.pt` (V4, proxy only)
Fall **classifier** (NOT loaded by the pipeline) on `data/crops/val`. imgsz=224.

```
yolo classify val model=models/fall_cls.pt data=data/crops imgsz=224 plots=True
```

top1=0.98738. Confusion (fallen positive): TP=101, FN=4, FP=0, TN=212 →
sensitivity=0.96190, FPR=0.00000 over 317 crops. This validates the unused
classifier at 224px crops, not the deployed `best.pt` fall-detection path, and
not at 1080p; it does not substantiate SR2.4 for the deployed system.
