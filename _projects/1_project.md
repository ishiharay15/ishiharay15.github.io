---
layout: page
title: Benchmarking Low-Light Image Enhancement (LLIE) and Restoration
description: Assessing effectiveness of Low-Light Image Enhancement as a pre-processing method for computer vision tasks
img: assets/img/1_project/enhancement_results.jpg
importance: 1
category: work
related_publications: true
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1_project/yolo_oppo_results.gif" title="Project Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Night-Vision Restoration! 🔦
</div>

# Project Overview:

This project focuses on enhancing image and video quality in low-light environments, particularly targeting the improvement of object detection in security camera footage. By addressing the challenges of lack of color and increased noise in low-light imagery, our solution aims to significantly enhance object recognition in dimly lit settings. Through the integration of noise reduction and color enhancement processes, coupled with object detection algorithms like YOLO, we aim to generate color-enhanced images with accurately detected object classes, thereby facilitating improved surveillance capabilities for security applications. To assess the efficacy of our methodology, we plan to employ metrics such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Mean Average Precision (mAP) to evaluate image quality improvements quantitatively. Additionally, we consider the compatibility of different image enhancement models with the YOLO object detection framework, ensuring optimal performance and accuracy in detecting objects within the reconstructed images. Our comprehensive approach aims to not only enhance image quality but also to optimize object detection capabilities, contributing to advancements in security surveillance effectiveness, and comprehension in low-light conditions.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1_project/pipeline.jpg" title="Project Pipeline" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Night-Vision Restoration Pipeline! 🌙
</div>

# Benchmarked Models:

**[Low-Light Image Enhancement with Multi-stage Residue Quantization (RQ-LLIE)](https://github.com/LiuYunlong99/RQ-LLIE):**
RQ-LLIE proposes a brightness-aware network for low-light image enhancement that uses normal-light priors to improve results. The method combines normal-light and low-light features and employs a brightness-aware attention module.\

**[Retinexformer](https://github.com/caiyuanhao1998/Retinexformer):**
The Retinexformer model is a One-stage Retinex-based Framework (ORF) for LLIE, using an Illumination-Guided Transformer (IGT) to model non-local interactions and restore image corruption.\

**[LLFormer](https://github.com/TaoWangzj/LLFormer):**
LLFormer is a transformer-based method that uses axis-based multi-head self-attention and cross-layer attention fusion to reduce complexity and improve performance.\

**[Global Structure-Aware Diffusion Process for Low-Light Image Enhancement (GSAD)](https://github.com/jinnh/GSAD):**
GSAD introduces a diffusion-based method for low-light image enhancement, incorporating curvature regularization and uncertainty-guided techniques to improve the enhancement process.

# Benchmarking Metrics:

**Peak Signal-to-Noise Ratio (PSNR):** PSNR quantifies image quality by comparing the original and enhanced images, focusing on greyscale intensity and RGB channels independently. It provides a measure of how closely the color and intensity of the two images align.\

**Structural Similarity Index (SSIM):** SSIM evaluates similarity in luminance, contrast, and structure between original and enhanced images. This metric offers a broader assessment of visual fidelity, often providing a more accurate representation than PSNR.\

**Mean Average Precision (mAP):** mAP calculates the average precision of model detections across all classes, using a defined accuracy threshold for object detection. This metric summarizes the model’s overall detection performance.\

# Results:

Our results demonstrate that the benchmarked models effectively enhance brightness, color, and reduce noise in low-light images, leading to improved object detection capabilities. By applying different enhancement techniques—RetinexFormer, Global Structure-Aware, LLFlow, and RQ-LLIE—each model showed distinct results in enhancing test images, leading to more accurate detection of objects using YOLOv5. Analysis of the [Exclusively Dark (ExDark) Image Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) reveals that the enhanced images enable YOLOv5 to detect objects with greater precision across various classes, compared to unenhanced images.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1_project/enhancement_results.jpg" title="Enhancement Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Night-Vision Restoration Results! 🔦
</div>

We also tested our models on the [Seeing Dynamic Scene in the Dark (SDSD)](https://github.com/dvlab-research/SDSD) dataset, which contains low-light videos, for further testing to validate the models' efficacy in dynamic, real-world scenarios. The results below highlights the potential of integrating low-light enhancement with object detection frameworks, promising advancements in security and surveillance applications.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1_project/yolo_oppo_results.gif" title="Video Enhancement Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    YOLO Video Test Results! 🎬
</div>
