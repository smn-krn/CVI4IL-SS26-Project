# Project Proposal

**Computer Vision for Interactive Learning (CVI4IL)**
**University Project – SS26**

## Project Title

**Estimating Animal Orientation in Aerial Thermal Imagery Using the BAMBI Dataset**

---

# 1. Introduction

Monitoring wildlife populations is an important task for ecological research, biodiversity conservation, and environmental management. Traditional wildlife monitoring methods often require manual observation or ground-based sensors, which can be time-consuming and limited in spatial coverage.

Recent advances in **unmanned aerial vehicles (UAVs)** and **computer vision** allow researchers to collect large amounts of aerial imagery and automatically analyze it using machine learning models. In particular, **thermal imaging from drones** makes it possible to detect animals even in low-light conditions or partially occluded environments such as forests.

The **BAMBI dataset** (Biodiversity Airborne Monitoring Based on Intelligent UAV Sampling), developed at FH Oberösterreich, provides annotated aerial thermal images of wildlife including deer and wild boar. While existing work primarily focuses on **detecting animals**, additional information such as **animal orientation** can significantly improve behavioral analysis and tracking.

This project aims to extend basic detection by estimating the **direction in which animals are facing**, using **head and tail keypoint annotations**.

---

# 2. Problem Statement

Current wildlife detection models can identify the **presence and location of animals** in aerial images, but they typically do not provide information about **animal orientation**.

Knowing the orientation of animals could enable:

* estimation of **movement direction**
* improved **multi-frame tracking**
* **behavior analysis**
* more accurate **wildlife monitoring**

To address this limitation, we propose a computer vision system that predicts two keypoints for each detected animal:

* **Head**
* **Tail**

From these keypoints, a **direction vector** can be calculated that indicates the orientation of the animal.

---

# 3. Project Objectives

The main objectives of this project are:

1. **Explore and preprocess the BAMBI dataset**
2. **Define a keypoint annotation format for head and tail positions**
3. **Develop a model capable of detecting these keypoints**
4. **Estimate the orientation of animals using predicted keypoints**
5. **Evaluate the accuracy of keypoint detection and orientation estimation**
6. **Visualize predictions and orientation vectors on aerial imagery**

---

# 4. Dataset

The project will use the **BAMBI dataset**, which contains aerial thermal imagery collected using UAV-mounted cameras.

### Dataset characteristics

* **Image type:** Thermal infrared images
* **Capture method:** UAV/drone flights
* **Environment:** Forest and open terrain

The dataset covers the following 12 classes:

| # | Species | Common Name | Wikidata                                                                                                  | Tracks | Key Frames |
|---|---|---|-----------------------------------------------------------------------------------------------------------|---:|---:|
| 1 | *Sus scrofa* | Wild boar | [Q58697](https://www.wikidata.org/wiki/Q58697)                                                            | 1,770 | 26,132 |
| 2 | *Cervus elaphus* | Red deer | [Q1219579](https://www.wikidata.org/wiki/Q1219579)                                                        | 1,613 | 26,674 |
| 3 | *Capreolus capreolus* | Roe deer | [Q122069](https://www.wikidata.org/wiki/Q122069)                                                          | 682 | 9,404 |
| 4 | *Dama dama* | Fallow deer | [Q20908334](https://www.wikidata.org/wiki/Q20908334)                                                      | 297 | 20,003 |
| 5 | *Capra ibex* | Alpine ibex | [Q168327](https://www.wikidata.org/wiki/Q168327)                                                          | 100 | 3,005 |
| 6 | *Rupicapra rupicapra* | Chamois | [Q131340](https://www.wikidata.org/wiki/Q131340)                                                          | 15 | 747 |
| 7 | *Aves* | Bird | [Q5113](https://www.wikidata.org/wiki/Q5113)                                                              | 75 | 942 |
| 8 | *Homo sapiens* | Human | [Q15978631](https://www.wikidata.org/wiki/Q15978631)                                                      | 93 | 1,158 |
| 9 | *Canis lupus familiaris* | Dog | [Q26972265](https://www.wikidata.org/wiki/Q26972265)                                                      | 7 | 95 |
| 10 | *Sus scrofa × Sus domesticus* | Hybrid pig | [Q602666](https://www.wikidata.org/wiki/Q602666) (no matching wikidata id, so workaround with forest hog) | 44 | 1,484 |
| 11 | — | No-animal | [Q10738](https://www.wikidata.org/wiki/Q10738) (it is not an animal, it is a rock, THE ROCK)              | 60 | 521 |
| 12 | — | Unknown | [Q24238356](https://www.wikidata.org/wiki/Q24238356)                                                      | 344 | 2,536 |

The dataset includes annotated bounding boxes for animals. For this project, additional **head and tail keypoints** will be used or created to allow orientation estimation.

---

# 5. Methodology

The project will be divided into several stages.

## 5.1 Data Preparation

* Load and explore the BAMBI dataset
* Analyze image characteristics and annotation formats
* Prepare a dataset split for training, validation, and testing
* Convert or generate head/tail annotations

## 5.2 Model Design

Two potential approaches could possibly be investigated:

### Keypoint Detection Model

Train a model that directly predicts **head and tail keypoints** for each animal.

Possible architectures include:

* keypoint detection CNNs
* pose estimation networks
* YOLO-based keypoint models

### Detection + Regression Approach

1. Detect animals using bounding boxes
2. Predict head/tail positions inside each bounding box

## 5.3 Orientation Estimation

Once head and tail keypoints are predicted, the orientation vector can be calculated:

$$
\vec{d} = \text{head} - \text{tail}
$$

This vector represents the **direction in which the animal is facing**.

## 5.4 Evaluation

Model performance will be evaluated using metrics such as:

* **Keypoint localization error**
* **Mean Euclidean distance between predicted and true keypoints**
* **Angular error of orientation vector**
* Detection accuracy (if detection is included)

## 5.5 Visualization

Results will be visualized by overlaying predictions on the images:

* predicted head and tail points
* orientation arrow
* bounding boxes (if applicable)

---

# 6. Expected Outcomes

At the end of the project we expect to achieve:

* a **data pipeline** for loading and preprocessing the BAMBI dataset
* a **trained model** capable of predicting head and tail keypoints
* a method to **estimate animal orientation**
* visualization tools for inspecting predictions
* quantitative evaluation of the model performance

---

# 7. Potential Challenges

Several challenges may arise during the project:

* Thermal images may have **low contrast or noise**
* Animals may appear **small or partially occluded**
* Head and tail positions may be **difficult to annotate accurately**
* Orientation prediction may be sensitive to small keypoint errors

These challenges will be addressed through preprocessing techniques, model design choices, and evaluation strategies.