# awesome-cvpr2022

workshop, tutorial, oral, and poster with notes in cvpr2022

**Wenhao(Reself) Chai**

*Undergraduate, UIUC*

![logo](https://cvpr2022.thecvf.com/sites/default/files/CVPR%20NewOrleansLogo_WebsiteCorner.png)


---
- [Workshops](#workshops)
  - [Machine Learning with Synthetic Data (SyntML) link](#machine-learning-with-synthetic-data-syntml-link)
  - [International Challenge on Activity Recognition (ActivityNet) link](#international-challenge-on-activity-recognition-activitynet-link)
  - [2nd Workshop and Challenge on Computer Vision in the Built Environment for the Design, Construction, and Operation of Buildings link](#2nd-workshop-and-challenge-on-computer-vision-in-the-built-environment-for-the-design-construction-and-operation-of-buildings-link)
  - [Workshop on Attention and Transformers in Vision link](#workshop-on-attention-and-transformers-in-vision-link)
  - [5th MUltimodal Learning and Applications Workshop (MULA) link](#5th-multimodal-learning-and-applications-workshop-mula-link)
  - [7th BMTT Workshop on Benchmarking Multi-Target Tracking: How Far Can Synthetic Data Take us? link](#7th-bmtt-workshop-on-benchmarking-multi-target-tracking-how-far-can-synthetic-data-take-us-link)
  - [L3D-IVU: Workshop on Learning with Limited Labelled Data for Image and Video Understanding link](#l3d-ivu-workshop-on-learning-with-limited-labelled-data-for-image-and-video-understanding-link)
- [Tutorials](#tutorials)
  - [Denoising Diffusion-based Generative Modeling: Foundations and Applications link](#denoising-diffusion-based-generative-modeling-foundations-and-applications-link)
  - [Recent Advances in Vision-and-Language Pre-training link](#recent-advances-in-vision-and-language-pre-training-link)
  - [Beyond Convolutional Neural Networks link](#beyond-convolutional-neural-networks-link)
  - [Evaluating Models Beyond the Textbook: Out-of-distribution and Without Labels link](#evaluating-models-beyond-the-textbook-out-of-distribution-and-without-labels-link)
- [Orals](#orals)
  - [Segmentation, Grouping and Shape Analysis](#segmentation-grouping-and-shape-analysis)
    - [1. Semantic-Aware Domain Generalized Segmentation link](#1-semantic-aware-domain-generalized-segmentation-link)
    - [2. Pointly-Supervised Instance Segmentation link](#2-pointly-supervised-instance-segmentation-link)
    - [3. Adaptive Early-Learning Correction for Segmentation From Noisy Annotations link](#3-adaptive-early-learning-correction-for-segmentation-from-noisy-annotations-link)
    - [4. Unsupervised Hierarchical Semantic Segmentation With Multiview Cosegmentation and Clustering Transformers link](#4-unsupervised-hierarchical-semantic-segmentation-with-multiview-cosegmentation-and-clustering-transformers-link)
  - [Video Analysis & Understanding](#video-analysis--understanding)
    - [5. Self-supervised Video Transformer link](#5-self-supervised-video-transformer-link)
    - [5. Dual-AI: Dual-Path Actor Interaction Learning for Group Activity Recognition link](#5-dual-ai-dual-path-actor-interaction-learning-for-group-activity-recognition-link)
  - [3D From Single Images](#3d-from-single-images)
    - [7. Tracking People by Predicting 3D Appearance, Location and Pose link](#7-tracking-people-by-predicting-3d-appearance-location-and-pose-link)
  - [Transfer / Low-Shot / Long-Tail Learning](#transfer--low-shot--long-tail-learning)
    - [8. OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization link](#8-ood-bench-quantifying-and-understanding-two-dimensions-of-out-of-distribution-generalization-link)
    - [9. Robust Fine-Tuning of Zero-Shot Models link](#9-robust-fine-tuning-of-zero-shot-models-link)
    - [10. Learning Distinctive Margin Toward Active Domain Adaptation link](#10-learning-distinctive-margin-toward-active-domain-adaptation-link)
    - [11. DINE: Domain Adaptation From Single and Multiple Black-Box Predictors link](#11-dine-domain-adaptation-from-single-and-multiple-black-box-predictors-link)
    - [12. Source-Free Object Detection by Learning To Overlook Domain Style link](#12-source-free-object-detection-by-learning-to-overlook-domain-style-link)
    - [13. Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization link](#13-exact-feature-distribution-matching-for-arbitrary-style-transfer-and-domain-generalization-link)
    - [14. Causality Inspired Representation Learning for Domain Generalization link](#14-causality-inspired-representation-learning-for-domain-generalization-link)
    - [15. Learning What Not To Segment: A New Perspective on Few-Shot Segmentation link](#15-learning-what-not-to-segment-a-new-perspective-on-few-shot-segmentation-link)
    - [16. Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation link](#16-towards-fewer-annotations-active-learning-via-region-impurity-and-prediction-uncertainty-for-domain-adaptive-semantic-segmentation-link)
    - [17. ADeLA: Automatic Dense Labeling With Attention for Viewpoint Shift in Semantic Segmentation link](#17-adela-automatic-dense-labeling-with-attention-for-viewpoint-shift-in-semantic-segmentation-link)
  - [Image & Video Synthesis and Generation](#image--video-synthesis-and-generation)
    - [18. Dataset Distillation by Matching Training Trajectories link](#18-dataset-distillation-by-matching-training-trajectories-link)
  - [Deep Learning Architectures & Techniques](#deep-learning-architectures--techniques)
    - [19. Controllable Dynamic Multi-Task Architectures link](#19-controllable-dynamic-multi-task-architectures-link)
  - [Human Pose Estimation & Tracking, Localization, and Object Pose Estimation](#human-pose-estimation--tracking-localization-and-object-pose-estimation)
    - [20. Temporal Feature Alignment and Mutual Information Maximization for Video-Based Human Pose Estimation link](#20-temporal-feature-alignment-and-mutual-information-maximization-for-video-based-human-pose-estimation-link)
    - [21. PoseTriplet: Co-Evolving 3D Human Pose Estimation, Imitation, and Hallucination Under Self-Supervision link](#21-posetriplet-co-evolving-3d-human-pose-estimation-imitation-and-hallucination-under-self-supervision-link)
    - [22. Generalizable Human Pose Triangulation link](#22-generalizable-human-pose-triangulation-link)
- [Posters](#posters)
  - [Segmentation, Grouping and Shape Analysis](#segmentation-grouping-and-shape-analysis-1)
    - [1. Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels link](#1-semi-supervised-semantic-segmentation-using-unreliable-pseudo-labels-link)
    - [2. Deep Hierarchical Semantic Segmentation link](#2-deep-hierarchical-semantic-segmentation-link)
    - [3. Amodal Segmentation Through Out-of-Task and Out-of-Distribution Generalization With a Bayesian Model link](#3-amodal-segmentation-through-out-of-task-and-out-of-distribution-generalization-with-a-bayesian-model-link)
    - [4. SWEM: Towards Real-Time Video Object Segmentation With Sequential Weighted Expectation-Maximization link](#4-swem-towards-real-time-video-object-segmentation-with-sequential-weighted-expectation-maximization-link)
    - [5. Accelerating Video Object Segmentation With Compressed Video link](#5-accelerating-video-object-segmentation-with-compressed-video-link)
    - [6. High Quality Segmentation for Ultra High-Resolution Images link](#6-high-quality-segmentation-for-ultra-high-resolution-images-link)
    - [7. Pin the Memory: Learning To Generalize Semantic Segmentation link](#7-pin-the-memory-learning-to-generalize-semantic-segmentation-link)
    - [8. Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity link](#8-open-world-instance-segmentation-exploiting-pseudo-ground-truth-from-learned-pairwise-affinity-link)
    - [9. Weakly Supervised Semantic Segmentation Using Out-of-Distribution Data link](#9-weakly-supervised-semantic-segmentation-using-out-of-distribution-data-link)
    - [10. Multimodal Material Segmentation link](#10-multimodal-material-segmentation-link)
    - [11. Semi-Supervised Learning of Semantic Correspondence With Pseudo-Labels link](#11-semi-supervised-learning-of-semantic-correspondence-with-pseudo-labels-link)
  - [Machine Learning](#machine-learning)
    - [12. A Re-Balancing Strategy for Class-Imbalanced Classification Based on Instance Difficulty link](#12-a-re-balancing-strategy-for-class-imbalanced-classification-based-on-instance-difficulty-link)
    - [13. How Much More Data Do I Need? Estimating Requirements for Downstream Tasks link](#13-how-much-more-data-do-i-need-estimating-requirements-for-downstream-tasks-link)
    - [14. Deep Safe Multi-view Clustering: Reducing the Risk of Clustering Performance Degradation Caused by View Increase link](#14-deep-safe-multi-view-clustering-reducing-the-risk-of-clustering-performance-degradation-caused-by-view-increase-link)
    - [15. Out-of-distribution Generalization with Causal Invariant Transformations link](#15-out-of-distribution-generalization-with-causal-invariant-transformations-link)
  - [Deep Learning Architectures & Techniques](#deep-learning-architectures--techniques-1)
    - [16. Single-Domain Generalized Object Detection in Urban Scene via Cyclic-Disentangled Self-Distillation link](#16-single-domain-generalized-object-detection-in-urban-scene-via-cyclic-disentangled-self-distillation-link)
    - [17. Revisiting Weakly Supervised Pre-Training of Visual Perception Models link](#17-revisiting-weakly-supervised-pre-training-of-visual-perception-models-link)
    - [18. Failure Modes of Domain Generalization Algorithms link](#18-failure-modes-of-domain-generalization-algorithms-link)
    - [19. Learning Part Segmentation Through Unsupervised Domain Adaptation From Synthetic Vehicles link](#19-learning-part-segmentation-through-unsupervised-domain-adaptation-from-synthetic-vehicles-link)
  - [Vision Applications & Systems](#vision-applications--systems)
    - [20. Large-Scale Pre-Training for Person Re-Identification With Noisy Labels link](#20-large-scale-pre-training-for-person-re-identification-with-noisy-labels-link)
  - [Recognition: Detection, Categorization, Retrieval](#recognition-detection-categorization-retrieval)
    - [21. Efficient Video Instance Segmentation via Tracklet Query and Proposal link](#21-efficient-video-instance-segmentation-via-tracklet-query-and-proposal-link)
    - [22. UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection [link]](#22-ubnormal-new-benchmark-for-supervised-open-set-video-anomaly-detection-link)
  - [3D From Single Images](#3d-from-single-images-1)
    - [23. Learning To Estimate Robust 3D Human Mesh From In-the-Wild Crowded Scenes link](#23-learning-to-estimate-robust-3d-human-mesh-from-in-the-wild-crowded-scenes-link)
    - [24. Exploiting Pseudo Labels in a Self-Supervised Learning Framework for Improved Monocular Depth Estimation link](#24-exploiting-pseudo-labels-in-a-self-supervised-learning-framework-for-improved-monocular-depth-estimation-link)
  - [Low-Level Vision](#low-level-vision)
    - [25. Multi-Scale Memory-Based Video Deblurring link](#25-multi-scale-memory-based-video-deblurring-link)
  - [Behavior Analysis](#behavior-analysis)
    - [25. Self-Supervised Keypoint Discovery in Behavioral Videos link](#25-self-supervised-keypoint-discovery-in-behavioral-videos-link)
    - [26. GLASS: Geometric Latent Augmentation for Shape Spaces link](#26-glass-geometric-latent-augmentation-for-shape-spaces-link)
  - [Vision & Language](#vision--language)
    - [27. Video-Text Representation Learning via Differentiable Weak Temporal Alignment link](#27-video-text-representation-learning-via-differentiable-weak-temporal-alignment-link)
    - [28. End-to-End Referring Video Object Segmentation With Multimodal Transformers link](#28-end-to-end-referring-video-object-segmentation-with-multimodal-transformers-link)
    - [29. Are Multimodal Transformers Robust to Missing Modality? link](#29-are-multimodal-transformers-robust-to-missing-modality-link)
    - [30. Robust Cross-Modal Representation Learning With Progressive Self-Distillation link](#30-robust-cross-modal-representation-learning-with-progressive-self-distillation-link)
    - [31. Multimodal Dynamics: Dynamical Fusion for Trustworthy Multimodal Classification link](#31-multimodal-dynamics-dynamical-fusion-for-trustworthy-multimodal-classification-link)
  - [Video Analysis & Understanding](#video-analysis--understanding-1)
    - [32. MLP-3D: A MLP-Like 3D Architecture With Grouped Time Mixing link](#32-mlp-3d-a-mlp-like-3d-architecture-with-grouped-time-mixing-link)
    - [33. Coarse-To-Fine Feature Mining for Video Semantic Segmentation link](#33-coarse-to-fine-feature-mining-for-video-semantic-segmentation-link)
    - [34. The DEVIL Is in the Details: A Diagnostic Evaluation Benchmark for Video Inpainting link](#34-the-devil-is-in-the-details-a-diagnostic-evaluation-benchmark-for-video-inpainting-link)
    - [35. YouMVOS: An Actor-Centric Multi-Shot Video Object Segmentation Dataset link](#35-youmvos-an-actor-centric-multi-shot-video-object-segmentation-dataset-link)
    - [36. Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark link](#36-large-scale-video-panoptic-segmentation-in-the-wild-a-benchmark-link)
  - [Transfer / Low-Shot / Long-Tail Learning](#transfer--low-shot--long-tail-learning-1)
    - [37. Which Model To Transfer? Finding the Needle in the Growing Haystack link](#37-which-model-to-transfer-finding-the-needle-in-the-growing-haystack-link)
    - [38. Task2Sim: Towards Effective Pre-Training and Transfer From Synthetic Data link](#38-task2sim-towards-effective-pre-training-and-transfer-from-synthetic-data-link)
  - [Pose Estimation & Tracking](#pose-estimation--tracking)
    - [39. MetaPose: Fast 3D Pose From Multiple Views Without 3D Supervision link](#39-metapose-fast-3d-pose-from-multiple-views-without-3d-supervision-link)
    - [40. Uncertainty-Aware Adaptation for Self-Supervised 3D Human Pose Estimation link](#40-uncertainty-aware-adaptation-for-self-supervised-3d-human-pose-estimation-link)
    - [41. PoseTrack21: A Dataset for Person Search, Multi-Object Tracking and Multi-Person Pose Tracking link](#41-posetrack21-a-dataset-for-person-search-multi-object-tracking-and-multi-person-pose-tracking-link)
    - [42. DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion link](#42-dancetrack-multi-object-tracking-in-uniform-appearance-and-diverse-motion-link)
    - [43. DiffPoseNet: Direct Differentiable Camera Pose Estimation link](#43-diffposenet-direct-differentiable-camera-pose-estimation-link)
  - [Recognition: Detection, Categorization, Retrieval](#recognition-detection-categorization-retrieval-1)
    - [44. Multi-Granularity Alignment Domain Adaptation for Object Detection link](#44-multi-granularity-alignment-domain-adaptation-for-object-detection-link)
    - [45. Cross-Domain Adaptive Teacher for Object Detection link](#45-cross-domain-adaptive-teacher-for-object-detection-link)
  - [Self-, Semi-, Meta-, & Unsupervised Learning](#self--semi--meta---unsupervised-learning)
    - [46. DASO: Distribution-Aware Semantics-Oriented Pseudo-Label for Imbalanced Semi-Supervised Learning link](#46-daso-distribution-aware-semantics-oriented-pseudo-label-for-imbalanced-semi-supervised-learning-link)
    - [47. Unbiased Teacher v2: Semi-Supervised Object Detection for Anchor-Free and Anchor-Based Detectors link](#47-unbiased-teacher-v2-semi-supervised-object-detection-for-anchor-free-and-anchor-based-detectors-link)
    - [48. Semi-Supervised Semantic Segmentation With Error Localization Network link](#48-semi-supervised-semantic-segmentation-with-error-localization-network-link)
    - [49. Debiased Learning From Naturally Imbalanced Pseudo-Labels link](#49-debiased-learning-from-naturally-imbalanced-pseudo-labels-link)
  - [Image & Video Synthesis and Generation](#image--video-synthesis-and-generation-1)
    - [50. Multi-View Consistent Generative Adversarial Networks for 3D-Aware Image Synthesis link](#50-multi-view-consistent-generative-adversarial-networks-for-3d-aware-image-synthesis-link)
  - [Datasets and Evaluation](#datasets-and-evaluation)
    - [51. SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation link](#51-shift-a-synthetic-driving-dataset-for-continuous-multi-task-domain-adaptation-link)
    - [52. Replacing Labeled Real-Image Datasets With Auto-Generated Contours link](#52-replacing-labeled-real-image-datasets-with-auto-generated-contours-link)

---


## Workshops

CVPR workshop June 19-20th.

[full schedule](https://cvpr2022.thecvf.com/workshop-schedule)


### Machine Learning with Synthetic Data (SyntML) [link](https://syntml-cvpr2022-workshop.github.io/)

Synthetic data are labeled data made using computer graphic. They are cheap, clean, and have richness of label. 
Keyword: *Domain mismatch*, *Diversity*

1. human synthesis *Google*
    - Procedural face generation\
       templete + features paradigm\
       features can be: identity, expression, pose
    - Hair and clothing
    - Environment
    - Render (Blender)

2. synthetic data & simulation *Nvidia*
    - graphics
        geometry + texture by a distribution
    - mixed reality
        *light* estimation + AR
    - generative models
        GAN / diffusion model

3. crossing the domain gap with synthetic data *Datagen*
    - why synthetic data?
      - pixel-accurate labels 
      - rich annotationns 
      - full control
    - types of gap
      - *photorealism gap*
      - pose gap
      - augmentation gap
      - annotation gap
    - styleGAN
      - cascade "parameter w class" (also like templete + features)
      - inversion / editing (good sensitivity)
    - mix sythetic data with real data (when limited) can achieve better performance
    - address domain gap
      - photorealism
      - label adaptation
      - add noise
      - global scene parameter distribution (lights, camera, pose)

### International Challenge on Activity Recognition (ActivityNet) [link](http://activity-net.org/challenges/2022/)

task: real-time online untrimmed security video action detection\
object: single / multi / interaction\
pipeline:
- detection
- background removal
- tracking (IOU-based)
- classification

related concept:
- domain adaptation
- overlapping spatio-temperal
- class-unbalance
- multi-label
- generalization performance

### 2nd Workshop and Challenge on Computer Vision in the Built Environment for the Design, Construction, and Operation of Buildings [link](https://cv4aec.github.io/)

- task: building model through point clouds to room map
- key tech: semantic segmentation of point clouds

### Workshop on Attention and Transformers in Vision [link](https://sites.google.com/view/t4v-cvpr22)

1. Visual Attention with Recurrency and Sparsity
2. BoxeR: Box-Attention for 2D and 3D Transformers
   - 2D / 3D object detection or segmentation
   - query: reference window
   - key: learnable relative region
   - multi-scale feature map
3. Depth Estimation with Simplified Transformers
   - FC -> 1x1 Conv.
4. M2F3D: MaskFormer fo 3D Instance Segmentation
   - top-down / bottom-up
   - sparse Conv.

### 5th MUltimodal Learning and Applications Workshop (MULA) [link](https://mula-workshop.github.io/)
Learning to Navigate from Vision and Language
- human use semantic priors to understand and navigate in unseen environment
- RL bottlenecks to progress on semantic navigation: scalability, diversity
- no need to learn a policy -> greedy

### 7th BMTT Workshop on Benchmarking Multi-Target Tracking: How Far Can Synthetic Data Take us? [link](https://motchallenge.net/workshops/bmtt2022/)

### L3D-IVU: Workshop on Learning with Limited Labelled Data for Image and Video Understanding [link](https://sites.google.com/view/l3d-ivu/)
- Low-Shot Scene Decomposition via Reconstruction
  - featurize 3D scene behind the image
  - fuse information form range sensors
  - RGB rendering is useful pre-training for detections
  - continues 3D feature maps with implicit functions
  - unsupervised detection: where and what, decouple these
  - unsupervised 3D segmentation via reconstruction loss

## Tutorials

CVPR tutorial June 19-20th.

### Denoising Diffusion-based Generative Modeling: Foundations and Applications [link](https://cvpr2022-tutorial-diffusion-models.github.io/)

- kinds of diffusion model
    - momentum-based
    - energy-based
    - latent-space (with pretrained VAE): faster  and simpler
    - distilation (merge steps)
    - discrete state diffusion model
- high-resolution
    - condition form: scalar / image / text
    - quality-diversity trade-off
    - cascade generation with super-resolution method
- application
    - semantic segmentation
    - image editing
    - adversarial robustness (purfied image)
    - video generation
      - types
        - all frames
        - past frames
        - future frames
        - interpolation
      - tips: training with different types of mask / use time position encodings to encode times
      - backbone: 3D Conv. / 2D Conv. + Att. (ignore initially when train)
      - long-term: generate a frame far away and then interpolation
    - medical imaging\
        reconstract original image from sparse measurements\
        high-level idea: learn pretrained on pure dataset momdel as "prior" than guide synthesis conditioned on sparse obvervations
    - 3D shape generation\
        through point clouds
- future trend
    - why diffusion models perform better?
    - how can we improve VAE / flow from diffusion model?
    - sampling from diffusion model is still slow
    - diffusion model can be considered as latent variable model *without semantic*, if with?
    - can diffusion model help to discrimination applications?
    - what are the best network architectures for diffusion model instead of UNet?
    - other data modality further than 2D image
    - controllable generation
    - in some application replace GAN with diffusion model


### Recent Advances in Vision-and-Language Pre-training [link](https://vlp-tutorial.github.io/2022/)

- unifying text and image
- avoiding explicit detection module
- high resolution computing cost
- coarse to fine two-stage VLP
- fusion in the backbone

### Beyond Convolutional Neural Networks [link](https://sites.google.com/view/cvpr-2022-beyond-cnn)

- DETR: DEtection TransfoRmer
  - idea: pose the task directly as set prediction, using a transformer encoder-decoder
  - bipartite match

### Evaluating Models Beyond the Textbook: Out-of-distribution and Without Labels [link](https://sites.google.com/view/evalmodel)

- robustness encompasses a broad range of phenomena (adv. examples, corruptions, nat. dist shift, etc.)
- some forms of robustness are currently orthogonal
- consistent trends across natural distribution shifts -> need more fine-grained understanding of different robustness notions.
- training data plays a key role in creating broadly robust models (e.g., CLIP). -> How do we construct training sets that enable broadly reliable models?
- very large improvements in OOD robustness

## Orals

### Segmentation, Grouping and Shape Analysis

#### 1. Semantic-Aware Domain Generalized Segmentation [link](https://arxiv.org/abs/2204.00822)

- sementic-aware normalization adapts a multi-branch normalization strategy, aiming to transform the input feature map into the category-level normalized features that are semantic-aware center aligned.

#### 2. Pointly-Supervised Instance Segmentation [link](https://arxiv.org/abs/2104.06404)

@Bowen Cheng

- training with pointed-based annotation
- implicit pointrend

#### 3. Adaptive Early-Learning Correction for Segmentation From Noisy Annotations [link](https://arxiv.org/abs/2110.03740)

- how to define early-training stage without ground truth?
- how to utilze noisy pesudo label?

#### 4. Unsupervised Hierarchical Semantic Segmentation With Multiview Cosegmentation and Clustering Transformers [link](https://arxiv.org/abs/2204.11432)

### Video Analysis & Understanding

#### 5. Self-supervised Video Transformer [link](https://arxiv.org/abs/2112.01514)

#### 5. Dual-AI: Dual-Path Actor Interaction Learning for Group Activity Recognition [link](https://arxiv.org/abs/2204.02148)

see the notes https://reself-c.github.io/DualAI 

### 3D From Single Images

#### 7. Tracking People by Predicting 3D Appearance, Location and Pose [link](https://arxiv.org/abs/2111.07868)

### Transfer / Low-Shot / Long-Tail Learning

#### 8. OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization [link](https://arxiv.org/abs/2106.03721)

- two dimensions of distribution shift
  - diversity shift -> shift in label
  - correlation shift -> shift in mapping

#### 9. Robust Fine-Tuning of Zero-Shot Models [link](https://arxiv.org/abs/2109.01903)

- weight-space ensemble of Fine-tune model and Zero-shot model (linear)

#### 10. Learning Distinctive Margin Toward Active Domain Adaptation [link](https://arxiv.org/abs/2203.05738)

- data sample strategy
  - classic uncertainty sample
  - diversity sample
  - multi-index evaluation
  - adversarial learning
  - ...
  - margin sample (this work)

#### 11. DINE: Domain Adaptation From Single and Multiple Black-Box Predictors [link](https://arxiv.org/abs/2104.01539)

- BB-SFDA: only logits

#### 12. Source-Free Object Detection by Learning To Overlook Domain Style [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Source-Free_Object_Detection_by_Learning_To_Overlook_Domain_Style_CVPR_2022_paper.pdf)

- augmentation + alignment

#### 13. Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization [link](https://arxiv.org/abs/2203.07740)

#### 14. Causality Inspired Representation Learning for Domain Generalization [link](https://arxiv.org/abs/2203.14237)

#### 15. Learning What Not To Segment: A New Perspective on Few-Shot Segmentation [link](https://arxiv.org/abs/2203.07615)

#### 16. Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation [link](https://arxiv.org/abs/2111.12940)

- pretrain + active-learning

#### 17. ADeLA: Automatic Dense Labeling With Attention for Viewpoint Shift in Semantic Segmentation [link](https://arxiv.org/abs/2107.14285)

- viewpoint change causes a prior shift for scene parsing

### Image & Video Synthesis and Generation

#### 18. Dataset Distillation by Matching Training Trajectories [link](https://arxiv.org/abs/2203.11932)

- compress the dataset from 50k to 10 by matching the parameter in the model

### Deep Learning Architectures & Techniques

#### 19. Controllable Dynamic Multi-Task Architectures [link](https://arxiv.org/abs/2203.14949)

- select the path and weight for a completed multi-task network architecture

### Human Pose Estimation & Tracking, Localization, and Object Pose Estimation

#### 20. Temporal Feature Alignment and Mutual Information Maximization for Video-Based Human Pose Estimation [link](https://arxiv.org/abs/2203.15227)

#### 21. PoseTriplet: Co-Evolving 3D Human Pose Estimation, Imitation, and Hallucination Under Self-Supervision [link](https://arxiv.org/abs/2203.15625)

#### 22. Generalizable Human Pose Triangulation [link](https://arxiv.org/abs/2110.00280)

## Posters

### Segmentation, Grouping and Shape Analysis

#### 1. Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels [link](https://arxiv.org/abs/2203.03884)

#### 2. Deep Hierarchical Semantic Segmentation [link](https://arxiv.org/abs/2203.14335)

#### 3. Amodal Segmentation Through Out-of-Task and Out-of-Distribution Generalization With a Bayesian Model [link](https://arxiv.org/abs/2010.13175)

- trained with bounding box and output is visible mask
- out-of-task and out-of-distribution generalization with a Bayesian generative model

#### 4. SWEM: Towards Real-Time Video Object Segmentation With Sequential Weighted Expectation-Maximization [link](https://openaccess.thecvf.com/content/CVPR2022/html/Lin_SWEM_Towards_Real-Time_Video_Object_Segmentation_With_Sequential_Weighted_Expectation-Maximization_CVPR_2022_paper.html)

- use point feature memory

#### 5. Accelerating Video Object Segmentation With Compressed Video [link](https://arxiv.org/abs/2107.12192)

- use residual between frames
- only inference on key frame and propagate the others by residual

#### 6. High Quality Segmentation for Ultra High-Resolution Images [link](https://arxiv.org/abs/2111.14482)

- calculate the relationship between the coordinate of low-resolution feature and ultra high-resolution target to get position information.

#### 7. Pin the Memory: Learning To Generalize Semantic Segmentation [link](https://arxiv.org/abs/2204.03609)

- store the feature as memory when inference on other domain
- close-set assumption, no label mismatch

#### 8. Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity [link](https://arxiv.org/abs/2204.06107)

- learn a pairwise affinity for each pixels
- a data augmentation strategy
- learn a binary and then classification (is that a object first?)

#### 9. Weakly Supervised Semantic Segmentation Using Out-of-Distribution Data [link](https://arxiv.org/abs/2203.03860)

#### 10. Multimodal Material Segmentation [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Multimodal_Material_Segmentation_CVPR_2022_paper.pdf)

- material segmentation (may close to texture but not so semantic)

#### 11. Semi-Supervised Learning of Semantic Correspondence With Pseudo-Labels [link](https://arxiv.org/pdf/2203.16038.pdf)

### Machine Learning

#### 12. A Re-Balancing Strategy for Class-Imbalanced Classification Based on Instance Difficulty [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_A_Re-Balancing_Strategy_for_Class-Imbalanced_Classification_Based_on_Instance_Difficulty_CVPR_2022_paper.pdf)

- resampling and reweighing for long-tail dataset
- class balance and hardness balance
- define a difficulty for classification

#### 13. How Much More Data Do I Need? Estimating Requirements for Downstream Tasks [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Mahmood_How_Much_More_Data_Do_I_Need_Estimating_Requirements_for_CVPR_2022_paper.pdf)

- estimate the amount of data needed
- most regession functions significantly over- or under- estimate how much data we needed

#### 14. Deep Safe Multi-view Clustering: Reducing the Risk of Clustering Performance Degradation Caused by View Increase [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Deep_Safe_Multi-View_Clustering_Reducing_the_Risk_of_Clustering_Performance_CVPR_2022_paper.pdf)

#### 15. Out-of-distribution Generalization with Causal Invariant Transformations [link](https://arxiv.org/abs/2203.11528)

### Deep Learning Architectures & Techniques

#### 16. Single-Domain Generalized Object Detection in Urban Scene via Cyclic-Disentangled Self-Distillation [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation_CVPR_2022_paper.pdf)

#### 17. Revisiting Weakly Supervised Pre-Training of Visual Perception Models [link](https://arxiv.org/abs/2201.08371)

- multi-label (hashtags) classification
- target is a uniform probality distribution on all hashtags for an image

#### 18. Failure Modes of Domain Generalization Algorithms [link](https://arxiv.org/abs/2111.13733)

#### 19. Learning Part Segmentation Through Unsupervised Domain Adaptation From Synthetic Vehicles [link](https://arxiv.org/abs/2103.14098)

### Vision Applications & Systems

#### 20. Large-Scale Pre-Training for Person Re-Identification With Noisy Labels [link](https://arxiv.org/abs/2203.16533)

### Recognition: Detection, Categorization, Retrieval

#### 21. Efficient Video Instance Segmentation via Tracklet Query and Proposal [link](https://arxiv.org/abs/2203.01853)

- both tracklet and appearance query
- both bounding box and mask output

#### 22. UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection [link]
(https://arxiv.org/abs/2111.08644)

### 3D From Single Images

#### 23. Learning To Estimate Robust 3D Human Mesh From In-the-Wild Crowded Scenes [link](https://arxiv.org/abs/2104.07300)

- use 2d pose to reduce domain gap
- self-updated 2d pose from off-the-shelf model

#### 24. Exploiting Pseudo Labels in a Self-Supervised Learning Framework for Improved Monocular Depth Estimation [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Petrovai_Exploiting_Pseudo_Labels_in_a_Self-Supervised_Learning_Framework_for_Improved_CVPR_2022_paper.pdf)

- augmentation + consistency

### Low-Level Vision

#### 25. Multi-Scale Memory-Based Video Deblurring [link](https://arxiv.org/abs/2204.02977)

- multi-scale 
- memory-based, remember the sharp and inference on blur

### Behavior Analysis

#### 25. Self-Supervised Keypoint Discovery in Behavioral Videos [link](https://arxiv.org/abs/2112.05121)

- self-supervised pretraining + downstream tasks

#### 26. GLASS: Geometric Latent Augmentation for Shape Spaces [link](https://arxiv.org/abs/2108.03225)

### Vision & Language

#### 27. Video-Text Representation Learning via Differentiable Weak Temporal Alignment [link](https://arxiv.org/abs/2203.16784)

- pretraining though multimodal alignment like video version CLIP

#### 28. End-to-End Referring Video Object Segmentation With Multimodal Transformers [link](https://arxiv.org/abs/2111.14821)

- multimodal transformer
- parallel for all the frames instead of sequetial based on memory bank

#### 29. Are Multimodal Transformers Robust to Missing Modality? [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.pdf)

#### 30. Robust Cross-Modal Representation Learning With Progressive Self-Distillation [link](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Andonian_Robust_Cross-Modal_Representation_CVPR_2022_supplemental.pdf)

#### 31. Multimodal Dynamics: Dynamical Fusion for Trustworthy Multimodal Classification [link](https://arxiv.org/abs/2204.00102)

### Video Analysis & Understanding

#### 32. MLP-3D: A MLP-Like 3D Architecture With Grouped Time Mixing [link](https://arxiv.org/abs/2206.06292)

#### 33. Coarse-To-Fine Feature Mining for Video Semantic Segmentation [link](https://arxiv.org/abs/2204.03330)

#### 34. The DEVIL Is in the Details: A Diagnostic Evaluation Benchmark for Video Inpainting [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Szeto_The_DEVIL_Is_in_the_Details_A_Diagnostic_Evaluation_Benchmark_CVPR_2022_paper.pdf)

#### 35. YouMVOS: An Actor-Centric Multi-Shot Video Object Segmentation Dataset [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_YouMVOS_An_Actor-Centric_Multi-Shot_Video_Object_Segmentation_Dataset_CVPR_2022_paper.pdf)

#### 36. Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark [link](https://openaccess.thecvf.com/content/CVPR2022/html/Miao_Large-Scale_Video_Panoptic_Segmentation_in_the_Wild_A_Benchmark_CVPR_2022_paper.html)

- changable reflective field for attention

### Transfer / Low-Shot / Long-Tail Learning

#### 37. Which Model To Transfer? Finding the Needle in the Growing Haystack [link](https://arxiv.org/abs/2010.06402)

- pretrain model selecting for downstream tasks

#### 38. Task2Sim: Towards Effective Pre-Training and Transfer From Synthetic Data [link](https://arxiv.org/abs/2112.00054)

- use RL to control the parameter of synthetic data generator

### Pose Estimation & Tracking

#### 39. MetaPose: Fast 3D Pose From Multiple Views Without 3D Supervision [link](https://arxiv.org/abs/2108.04869)

#### 40. Uncertainty-Aware Adaptation for Self-Supervised 3D Human Pose Estimation [link](https://arxiv.org/abs/2203.15293)

#### 41. PoseTrack21: A Dataset for Person Search, Multi-Object Tracking and Multi-Person Pose Tracking [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Doring_PoseTrack21_A_Dataset_for_Person_Search_Multi-Object_Tracking_and_Multi-Person_CVPR_2022_paper.pdf)

#### 42. DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion [link](https://arxiv.org/abs/2111.14690)

- train a refine net with only 2D gt

#### 43. DiffPoseNet: Direct Differentiable Camera Pose Estimation [link](https://arxiv.org/abs/2203.11174)


### Recognition: Detection, Categorization, Retrieval

#### 44. Multi-Granularity Alignment Domain Adaptation for Object Detection [link](https://arxiv.org/abs/2203.16897s)

#### 45. Cross-Domain Adaptive Teacher for Object Detection [link](https://arxiv.org/abs/2111.13216)

- pixel-/instance-/catagory- level discrimination
 
### Self-, Semi-, Meta-, & Unsupervised Learning

#### 46. DASO: Distribution-Aware Semantics-Oriented Pseudo-Label for Imbalanced Semi-Supervised Learning [link](https://arxiv.org/abs/2106.05682)

#### 47. Unbiased Teacher v2: Semi-Supervised Object Detection for Anchor-Free and Anchor-Based Detectors [link](https://arxiv.org/abs/2206.09500)

#### 48. Semi-Supervised Semantic Segmentation With Error Localization Network [link](https://arxiv.org/abs/2204.02078s)

#### 49. Debiased Learning From Naturally Imbalanced Pseudo-Labels [link](https://arxiv.org/abs/2201.01490)

- similar to entropy filter

### Image & Video Synthesis and Generation

#### 50. Multi-View Consistent Generative Adversarial Networks for 3D-Aware Image Synthesis [link](https://arxiv.org/abs/2204.06307)

### Datasets and Evaluation

#### 51. SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation [link](https://arxiv.org/abs/2206.08367)

#### 52. Replacing Labeled Real-Image Datasets With Auto-Generated Contours [link](https://openaccess.thecvf.com/content/CVPR2022/html/Kataoka_Replacing_Labeled_Real-Image_Datasets_With_Auto-Generated_Contours_CVPR_2022_paper.html)
