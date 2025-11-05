# mobile-leafdoc
We are working to reproduce the Frontiers in Plant Science (Dec 2023) model that fine-tunes **MobileNetV3-Small** on **PlantVillage** and then performs **post-training quantization** to an **ONNX** network. The paper (https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1308528/full) reports ~**99.5% test accuracy** on PlantVillage and a parameter reduction from ~**1.5M → 0.93M** with no accuracy loss after quantization, highlighting suitability for mobile/edge deployment.
However, lab-style PlantVillage (https://github.com/spMohanty/PlantVillage-Dataset/tree/master) images may not reflect real field conditions, prior work cited in the paper shows that performance can degrade on in-the-wild photos with shadows, clutter, and misalignment.

*We therefore target **generalization and reliability** under real-world shift while preserving edge constraints (size/latency) using the PlantDoc dataset (https://github.com/pratikkayal/PlantDoc-Dataset)*

## Experimental design

PlantVillage: ~54k leaf images, 38 classes, studio-like (plain background). Great for fast training but not realistic.
PlantDoc: ~2.6k “in-the-wild” images (phones/internet), 13 species, ~17 classes.

### ✅ baseline:

PV only → MobilenetV3-Small, paper settings. Measure ID metrics.

Major Gaps:
1. No preprocessing done on the PlantVillage dataset
2. Trained on only the PlantVillage dataset

> Runs
> 
> 
> ### Frontiers2023/Run 1
> 
> epochs 20
> ran on plantvillage dataset, no normalization
> exact settings as given on the paper
> 
> ### Frontiers2023/Run 2
> 
> epochs 200
> ran on plantvillage dataset, no normalization
> exact settings as given on the paper
> 
> - got acc ~ 99.6%, slightly higher than the paper
> - this is the final checkpoint
✅  Done

### ⭕ robustness block:

Experimenting on 3 levels

✅  PV only → test on PD (zero-shot)
    
    train on PlantVillage only, test on PlantDoc.
    
- [ ]  PV → fine-tune on PD
    
    train on PlantVillage, then fine-tune on PlantDoc.
    
- [ ]  PV+PD mixed (balanced sampler) with color constancy + RandAugment.
    1. train on PlantVillage+PlantDoc (only overlapping classes)
    2. using a dataset-balanced sampler and robust augmentations.

### ⬜ check compression block:

 - first perform PTQ-INT8
 - Compare FP32 vs INT8 on ID + OOD and report edge metrics.
