# Computer Vision Project — Working with the BAMBI dataset
CVI4IL – University Project (Deadline: Currently unknown)

---

# Team
- Annalena Salchegger
- Celina Binder
- Simone Kern

Distribution and tasks handled per member will be decided on a weekly basis, depending on the project progress and individual preferences, and will be added in this section as we go.

---

# Project Overview

This repository contains the full workflow for our computer vision project using the **BAMBI wildlife drone dataset** provided by FH Oberösterreich.

The goal of this project is to **detect animals in aerial thermal images and determine their orientation** by marking the **head and tail positions**.  
From these two keypoints we can estimate the **direction the animal is facing**.

Understanding the orientation of animals can be useful for:

- behavioral analysis
- movement prediction
- wildlife monitoring
- improved tracking across frames

The project focuses on building a **computer vision pipeline** that processes drone imagery and predicts **two keypoints per animal:**

- **Head**
- **Tail**

From these points we can derive a **direction vector** representing the animal's orientation.

---

# Project Objectives

The main objectives of this project are:

1. **Work with the BAMBI dataset**
2. **Prepare and preprocess aerial thermal imagery**
3. **Annotate or use annotations for head and tail positions**
4. **Train a model capable of detecting these keypoints**
5. **Evaluate prediction accuracy**
6. **Visualize predicted orientation vectors**

---

# Repository Structure

```

CVI4IL-SS26-Project
│
├── code                       # Structured into the corresponding tasks
│
├── data                       # Both original and processed
│
├── docs                       # Documentation and reports; contains proposal
│
├── pyproject.toml             # Python project configuration
├── uv.lock                    # Dependency lockfile
├── .python-version            # Python version specification
├── .gitignore
└── README.md

```

---

# Dataset — BAMBI

Links:

- https://zenodo.org/records/18772136
- https://github.com/bambi-eco/Dataset

The **BAMBI dataset** (Biodiversity Airborne Monitoring Based on Intelligent UAV Sampling) is a wildlife monitoring dataset collected using **drone-mounted thermal cameras**.

It contains aerial images and videos of animals in natural habitats such as forests and fields.

### Characteristics

- **Sensor type:** Thermal infrared camera
- **Platform:** UAV / drone
- **Environment:** Forest and open terrain
  
The dataset is typically used for tasks such as:

- wildlife detection
- animal tracking
- biodiversity monitoring
- aerial computer vision

### Species

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

### Annotation Format (MOT)

Annotations are stored as CSV files (no header) following a custom MOT format:

```
frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility, species, gender, age, is_propagated
```

| Column | Type | Description |
|---|---|---|
| `frame` | int | Frame index |
| `track_id` | int | Unique track identifier |
| `bb_left` | int | Bounding box left coordinate (px) |
| `bb_top` | int | Bounding box top coordinate (px) |
| `bb_width` | int | Bounding box width (px) |
| `bb_height` | int | Bounding box height (px) |
| `conf` | float | Confidence score |
| `class_id` | int | Species class identifier |
| `visibility` | float | Visibility (1.0 = fully visible, 0.0 = fully occluded) |
| `species` | str | Species name |
| `gender` | int | Gender (0 = unknown, 1 = male, 2 = female) |
| `age` | int | Age (0 = unknown, 1 = juvenile, 2 = adult) |
| `is_propagated` | int | 0 = annotated key frame, 1 = interpolated |

Class_ids are defined as `<wikidata_id>-<gender>-<age>-<visibility>` with the following mapping:
```json
{
  "class_mapping": {
    "Q10738-0-0-0": 0,
    "Q10738-0-0-1": 1,
    "Q1219579-0-0-0": 2,
    "Q1219579-0-0-1": 3,
    "Q1219579-0-1-0": 4,
    "Q1219579-0-1-1": 5,
    "Q1219579-0-2-0": 6,
    "Q1219579-0-2-1": 7,
    "Q1219579-1-0-0": 8,
    "Q1219579-1-0-1": 9,
    "Q1219579-1-2-0": 10,
    "Q1219579-1-2-1": 11,
    "Q1219579-2-0-0": 12,
    "Q1219579-2-0-1": 13,
    "Q1219579-2-1-0": 14,
    "Q1219579-2-2-0": 15,
    "Q1219579-2-2-1": 16,
    "Q122069-0-0-0": 17,
    "Q122069-0-0-1": 18,
    "Q122069-0-1-0": 19,
    "Q122069-0-1-1": 20,
    "Q122069-0-2-0": 21,
    "Q122069-0-2-1": 22,
    "Q131340-0-0-0": 23,
    "Q131340-0-0-1": 24,
    "Q131340-0-2-0": 25,
    "Q131340-0-2-1": 26,
    "Q15978631-0-0-0": 27,
    "Q15978631-0-0-1": 28,
    "Q15978631-0-2-0": 29,
    "Q15978631-0-2-1": 30,
    "Q15978631-2-0-0": 31,
    "Q15978631-2-2-0": 32,
    "Q15978631-2-2-1": 33,
    "Q168327-0-0-0": 34,
    "Q168327-0-0-1": 35,
    "Q168327-0-1-0": 36,
    "Q168327-0-1-1": 37,
    "Q168327-0-2-0": 38,
    "Q168327-0-2-1": 39,
    "Q168327-2-2-0": 40,
    "Q168327-2-2-1": 41,
    "Q20908334-0-0-0": 42,
    "Q20908334-0-1-0": 43,
    "Q20908334-1-0-0": 44,
    "Q20908334-1-1-0": 45,
    "Q20908334-1-2-0": 46,
    "Q20908334-2-2-0": 47,
    "Q20908334-2-2-1": 48,
    "Q24238356-0-0-0": 49,
    "Q24238356-0-0-1": 50,
    "Q24238356-0-2-0": 51,
    "Q24238356-0-2-1": 52,
    "Q26972265-0-0-0": 53,
    "Q26972265-0-2-0": 54,
    "Q26972265-0-2-1": 55,
    "Q5113-0-0-0": 56,
    "Q5113-0-0-1": 57,
    "Q5113-0-2-0": 58,
    "Q5113-0-2-1": 59,
    "Q58697-0-0-0": 60,
    "Q58697-0-0-1": 61,
    "Q58697-0-1-0": 62,
    "Q58697-0-1-1": 63,
    "Q58697-0-2-0": 64,
    "Q58697-0-2-1": 65,
    "Q58697-1-2-0": 66,
    "Q58697-1-2-1": 67,
    "Q58697-2-2-0": 68,
    "Q58697-2-2-1": 69,
    "Q602666-0-0-0": 70,
    "Q602666-0-0-1": 71,
    "Q602666-0-1-0": 72,
    "Q602666-0-2-0": 73,
    "Q602666-0-2-1": 74
  }
}
```

---

# Task Definition

In this project we extend standard animal detection by estimating **animal orientation**.

Each detected animal will have two keypoints:

- **Head**
- **Tail**

From these two points we compute:

```

orientation_vector = head_position - tail_position

```

This vector indicates **the direction the animal is facing**.

Example:

```

tail ●──────────▶ ● head
direction of movement / gaze

```

This approach allows us to determine:

- movement direction
- attention direction
- improved multi-frame tracking

---

# Methodology

The project pipeline consists of several steps:

### 1. Data Preparation
- Load BAMBI images
- Clean and structure the dataset
- Create or import annotations

### 2. Annotation Processing
- Store head/tail annotations
- Convert them into training format

### 3. Model Training
Possible approaches include:

- Keypoint detection networks
- Pose estimation models
- Object detection + keypoint regression

### 4. Evaluation
Evaluate predictions using metrics such as:

- Keypoint localization error
- Orientation angle error
- Detection accuracy

### 5. Visualization
Overlay predictions on images:

- predicted head position
- predicted tail position
- direction arrow

---

# Setup Instructions

## Clone the repository
```bash
git clone https://github.com/smn-krn/CVI4IL-SS26-Project.git
cd CVI4IL-SS26-Project/
```

## Project Dependencies & Environment Setup (using uv)

This project uses `uv` + `pyproject.toml` to manage Python dependencies.
Unlike pip or conda, uv supports dependency groups, which allow us to install only what we need for each part of the project (baseline models, ML models, neural models, visualizations, etc.).

This keeps environments small, fast, conflict-free, and makes collaboration easier.

------------------------------------------------------------
1. Installing Dependencies
------------------------------------------------------------

Install only the base dependencies (recommended for general users):

    uv sync

This installs only what is defined under:

    [project]
    dependencies = [...]

------------------------------------------------------------
Install dependencies for a specific model family
------------------------------------------------------------

Visualizations:

    uv sync --group viz

------------------------------------------------------------
Install multiple dependency groups example
------------------------------------------------------------

    uv sync --group ml --group neural

------------------------------------------------------------
Install everything (not recommended unless necessary)
------------------------------------------------------------

    uv sync --all-groups

------------------------------------------------------------
1. Adding New Dependencies
------------------------------------------------------------

Dependencies are tracked inside pyproject.toml.

Add a package to the base dependencies:

    uv add pandas

Add a package to a specific group:

Example: add Prophet to stats models:

    uv add prophet --group stats

Example: add TorchMetrics to neural models:

    uv add torchmetrics --group neural

Example: add seaborn extensions to viz tools:

    uv add seaborn ruff --group viz

------------------------------------------------------------
1. Checking Installed Packages
------------------------------------------------------------

List everything currently installed:

    uv pip list

See which groups are available:

    uv tree

------------------------------------------------------------
Summary Table
------------------------------------------------------------

Task                                 | Command
------------------------------------ | -----------------------------------------------
Install base deps                    | uv sync
Install viz tools                    | uv sync --group viz
Add package to base                  | uv add PACKAGE
Add package to a group               | uv add PACKAGE --group GROUPNAME
Install multiple groups              | uv sync --group GROUP1 --group GROUP2

------------------------------------------------------------
Groups
------------------------------------------------------------

name displayed here is actual group name; below are the packages installed per group

base                                 | viz
------------------------------------ | -----------------------------------------------
matplotlib                    | plotly
numpy                    | 
pandas                  | 
scikit-learn               | 
seaborn              | 
ipykernel |


------------------------------------------------------------


We are using Python **>=3.12**

**That is all you need to do for the setup!**


# Branch Instructions

Make sure you are in the project folder.

## Check current branch
```bash
git branch
```
 should be *main and maybe more.

if you're not on main (so if * is on a different branch), switch via 
```bash 
git checkout main
```

## Pull the latest changes from origin
```bash
git pull origin main
```

## Create a new feature branch
```bash
git checkout -b <feature-name>
```

-b means: create and switch to the new branch.

**Naming convention for branch; NOT for commit message:** 

[FirstLetterOfFirstName][FirstLetterOfLastName]-[FeatureAdded]


EXAMPLE: CB-Preprocessing or AS-Visualizations

## First push
```bash
git push -u origin <feature-name> 
```

Do this before any changes, because the very first push requires special handling using the "-u origin" feature. 

## All further pushes

After this, you can treat it normally using 
```bash 
git add . 
git commit -m "Message"
git push
```

## Merging

Merge via github website

1. Go to the repo on GitHub
2. Open the Pull Requests tab
3. GitHub usually suggests one automatically

If it doesn't 

1. Go to the repository, tab ```Pull Requests```
2. It should suggest one but if it doesn't click ```new pull request```
   
   ```base: main <- compare: [your-feature-branch]```

Should be able to switch without merging conflicts. If you have merging conflicts... pray.

## Switching to main

We first need to pull this from the repo so that we have this merge locally.

```bash
git fetch
git pull
```

now we can switch to the main

```bash
git checkout main
git pull origin main
```