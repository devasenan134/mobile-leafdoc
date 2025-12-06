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
- ✅  Done

### ✅ robustness block:

Experimenting on 3 levels

- ✅  PV only → test on PD (zero-shot)
    
    train on PlantVillage only, test on PlantDoc.
    
- ✅  PV → fine-tune on PD
    
    train on PlantVillage, then fine-tune on PlantDoc.
    
    Using 'Frontiers2023/Run 2/mobilenetv3small_best' model and fine-tune it with PlantDoc dataset. Fine-tune on PlantDoc dataset, with normalization and data augmentation. (run1 and run2 are not much different)
    > Runs
    > 
    > ### finetune/Run 1
    > 
    > epochs 20
    > - got best acc ~ 50%
    > 
    > ### finetune/Run 2
    > 
    > epochs 50
    > - got best acc ~ 53%
    > - evaluate on PlantDoc test data
    >       - Baseline (Zero-Shot): 2.54%
    >       - Fine-Tuned:  50.42%

    
- ✅ PV+PD mixed (balanced sampler) with color constancy + RandAugment.
    1. train on PlantVillage+PlantDoc (only overlapping classes)
    > ### mixed:
    > 
    > epochs 30
    > - PV test  acc=0.9964  F1=0.9954
    > - PD test  acc=0.5297  F1=0.5182

    
    2. using a dataset-balanced sampler and robust augmentations.

    > Runs
    > use RandAugment + ColorConstancy (with balance classes)
    > 
    > ### mixed_pv_pd/Run 1
    > 
    > epochs 10
    > - got best acc ~ 98%
    > 
    > ### mixed_pv_pd/Run 2
    > 
    > epochs 15
    > - got best acc ~ 99.4%
    > - evaluate on PlantDoc test data
    >       - Mixed:  56.78%

    3. Using weights for the PD data samples (to balance the datasize imbalance)
    > ### mixed_pdweighted:
    > 
    > epochs 30
    > - use RandAugment + ColorConstancy (with balance classes)
    > - PV test  acc=0.9974  F1=0.9965
    > - PD test  acc=0.5847  F1=0.5818  (topk_acc=0.86)
    > 
    4. try using other augmentation, Asymmetric (Heavy noise for PV, Light for PD), + FDA (Fourier Domain Adaptation)
    > Runs
    > use Asymmetric + FDA (with balance classes + domains (upweights PD 5x))
    > 
    > ### mixed_with_fda/Run 1
    > 
    > epochs 20 
    > - PV test: 98.6%
    > - PD test:  60.17%, (top k):  84.75% 
    >
    4. try using other augmentation, Asymmetric (Heavy noise for PV, Light for PD), + FDA (Fourier Domain Adaptation) with ImageNetV3_large
    > Runs
    > use ImageNetV3_large model
    > 
    > ### mixed_with_imageNewV3_large
    > 
    > epochs 10 
    > - PV test:  99.01%
    > - PD test:  61.44% 

### ❌ check compression block:

 - first perform PTQ-INT8
 - Compare FP32 vs INT8 on ID + OOD and report edge metrics.
