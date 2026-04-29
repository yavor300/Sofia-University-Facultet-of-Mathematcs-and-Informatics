# GutBrainIE @ CLEF 2026

Repository template and task documentation for participating in **GutBrainIE @ CLEF 2026**, Task #6 of the **BioASQ Lab 2026**. The challenge focuses on biomedical information extraction from PubMed titles and abstracts related to the gut microbiota and its connections with neurological and mental health conditions.

> **Challenge theme:** extract structured biomedical knowledge about the gut-brain interplay from scientific abstracts.

---

## Table of Contents

- [Overview](#overview)
- [Challenge Tasks](#challenge-tasks)
  - [Subtask 6.1.1: Named Entity Recognition](#subtask-611-named-entity-recognition-ner)
  - [Subtask 6.1.2: Named Entity Recognition and Disambiguation](#subtask-612-named-entity-recognition-and-disambiguation-nerd)
  - [Subtask 6.2.1: Mention-Level Relation Extraction](#subtask-621-mention-level-relation-extraction-m-re)
  - [Subtask 6.2.2: Concept-Level Relation Extraction](#subtask-622-concept-level-relation-extraction-c-re)
- [Biomedical Scope](#biomedical-scope)
- [Datasets](#datasets)
- [Dataset Format](#dataset-format)
- [Entity Labels](#entity-labels)
- [Relation Labels](#relation-labels)
- [Dataset Statistics](#dataset-statistics)
- [Evaluation](#evaluation)
- [Baseline Results](#baseline-results)
- [Submission Requirements](#submission-requirements)
- [Example Submission Formats](#example-submission-formats)
- [Recommended Repository Structure](#recommended-repository-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Important Dates](#important-dates)
- [References](#references)

---

## Overview

GutBrainIE @ CLEF 2026 is a Natural Language Processing challenge centered on **Information Extraction (IE)** from biomedical literature. Participants are given PubMed titles and abstracts discussing the relationship between the gut microbiota and conditions such as:

- Alzheimer's disease
- Parkinson's disease
- Multiple Sclerosis
- Amyotrophic Lateral Sclerosis
- Mental health conditions

The goal is to develop systems that extract structured entities and relations so that domain experts can better study the gut-brain axis.

The challenge contains two main task families:

1. **Entity extraction tasks**
   - Identify biomedical entity mentions.
   - Classify them into predefined categories.
   - Optionally link them to biomedical concept identifiers.

2. **Relation extraction tasks**
   - Identify whether a specific relationship holds between two entities or concepts.
   - Extract relation predicates between mentions or normalized concepts.

---

## Challenge Tasks

### Subtask 6.1.1: Named Entity Recognition (NER)

Participants must identify and classify biomedical entity mentions in PubMed titles and abstracts.

Each entity mention is represented as:

```text
(entityCategory ; entityLocation ; startOffset ; endOffset)
```

Required fields:

- `label`: entity category
- `location`: either `title` or `abstract`
- `start_idx`: character start offset
- `end_idx`: character end offset
- `text_span`: extracted entity text span

No concept URI is required for this subtask.

---

### Subtask 6.1.2: Named Entity Recognition and Disambiguation (NERD)

Participants must identify and classify entity mentions as in NER, but must also link each entity to a concept identifier from a biomedical reference resource.

Each entity mention is represented as:

```text
(entityCategory ; entityLocation ; startOffset ; endOffset ; conceptURI)
```

Required fields:

- `label`
- `location`
- `start_idx`
- `end_idx`
- `text_span`
- `uri`

This task evaluates both span detection/classification and concept normalization.

---

### Subtask 6.2.1: Mention-Level Relation Extraction (M-RE)

Participants must identify relations between specific entity mentions within a document.

Each relation is represented as:

```text
(subjectMention ; relationPredicate ; objectMention)
```

Required fields:

- `subject_text_span`
- `subject_label`
- `predicate`
- `object_text_span`
- `object_label`

The relation is evaluated at the mention level, meaning that the system should recover the relation between textual mentions appearing in the title or abstract.

---

### Subtask 6.2.2: Concept-Level Relation Extraction (C-RE)

Participants must identify relations between normalized biomedical concepts rather than surface text mentions.

Each relation is represented as:

```text
(subjectConceptURI ; subjectCategory ; relationPredicate ; objectConceptURI ; objectCategory)
```

Required fields:

- `subject_uri`
- `subject_label`
- `predicate`
- `object_uri`
- `object_label`

This task abstracts away from lexical variation and evaluates whether the system can recover knowledge-level relations between concepts.

---

## Biomedical Scope

The challenge focuses on biomedical literature related to the gut-brain interplay. Documents are PubMed article titles and abstracts, and the target knowledge includes entities such as bacteria, chemicals, diseases, food, drugs, microbiome concepts, genes, humans, animals, anatomical locations, and biomedical/statistical techniques.

---

## Datasets

The official datasets are available after registration for GutBrainIE @ CLEF 2026 / BioASQ Lab 2026.

The training data is divided into four main collections:

| Collection | Description |
|---|---|
| **Gold Collection** | Expert-curated annotations, primarily created by 7 expert annotators from the University of Padua with additional external expert contributors. |
| **Silver Collection** | Weakly curated annotations produced by trained students of Linguistics and Terminology under expert supervision. |
| **Silver Collection 2025** | Weakly curated data from the 2025 edition, with automatically generated concept-level annotations. |
| **Bronze Collection** | Distantly supervised annotations generated automatically. No manual revision was performed. |

The test set is a held-out selection of documents from the gold collection and contains only:

- PubMed ID
- title
- abstract

Participants must generate the required predictions for the selected subtask.

---

## Dataset Format

Annotations are provided in **JSON** format. Each entry corresponds to a PubMed article identified by its PMID.

A typical annotated entry may contain the following high-level fields:

```text
Metadata
Entities
Relations
Mention-level Relations
Concept-level Relations
```

### Metadata

Contains article-related information such as:

- title
- authors
- journal
- publication year
- abstract
- annotator identifier

Annotator identifiers may indicate expert, student, or distant annotations. This can be useful for weighting examples during training.

### Entities

Each entity object may include:

| Field | Description |
|---|---|
| `start_idx` | Character start offset of the entity mention. |
| `end_idx` | Character end offset of the entity mention. |
| `location` | Whether the entity occurs in the `title` or `abstract`. |
| `text_span` | Surface form of the entity mention. |
| `label` | Entity category. |
| `uri` | Concept URI from a biomedical reference resource. Required for NERD. |

### Relations

Each relation object may include:

| Field | Description |
|---|---|
| `subject_start` / `subject_end` | Character offsets of the subject mention. |
| `subject_location` | Location of the subject mention. |
| `subject_text_span` | Text span of the subject. |
| `subject_uri` | Concept URI of the subject. |
| `subject_label` | Entity label of the subject. |
| `predicate` | Relation label. |
| `object_start` / `object_end` | Character offsets of the object mention. |
| `object_location` | Location of the object mention. |
| `object_text_span` | Text span of the object. |
| `object_uri` | Concept URI of the object. |
| `object_label` | Entity label of the object. |

### Alternative Formats

The dataset is also provided in CSV or tabular formats. In those formats, the following files are separated:

- metadata file
- entities file
- relations file
- mention-level relations file
- concept-level relations file

CSV files use the pipe symbol `|` as separator, while tabular files use the tab character `\t`.

---

## Entity Labels

The challenge uses 13 predefined entity categories:

| Entity Label | Description |
|---|---|
| `anatomical location` | Named locations of or within the body. |
| `animal` | Non-human living organisms capable of voluntary movement. |
| `biomedical technique` | Techniques related to biological and physiological principles applied to clinical medicine. |
| `bacteria` | Bacterial organisms, including unicellular prokaryotic microorganisms. |
| `chemical` | Chemical substances, including metabolites and neurotransmitters. |
| `dietary supplement` | Products intended to increase nutrient intake. |
| `DDF` | Disease, Disorder, or Finding. |
| `drug` | Substances that may modify biological functions, especially therapeutic substances. |
| `food` | Substances consumed for nutritional purposes. |
| `gene` | Functional units of heredity. |
| `human` | Members of the species *Homo sapiens*. |
| `microbiome` | Microorganisms, their genomes, and their surrounding environmental conditions. |
| `statistical technique` | Methods for calculating, analyzing, or representing statistical data. |

---

## Relation Labels

The challenge defines relation predicates over valid head and tail entity types.

| Head Entity | Tail Entity | Predicate |
|---|---|---|
| Anatomical Location | Human / Animal | `located in` |
| Bacteria | Bacteria / Chemical / Drug | `interact` |
| Bacteria | DDF | `influence` |
| Bacteria | Gene | `change expression` |
| Bacteria | Human / Animal | `located in` |
| Bacteria | Microbiome | `part of` |
| Chemical | Anatomical Location / Human / Animal | `located in` |
| Chemical | Chemical | `interact`, `part of` |
| Chemical | Microbiome | `impact`, `produced by` |
| Chemical / Dietary Supplement / Drug / Food | Bacteria / Microbiome | `impact` |
| Chemical / Dietary Supplement / Food | DDF | `influence` |
| Chemical / Dietary Supplement / Drug / Food | Gene | `change expression` |
| Chemical / Dietary Supplement / Drug / Food | Human / Animal | `administered` |
| DDF | Anatomical Location | `strike` |
| DDF | Bacteria / Microbiome | `change abundance` |
| DDF | Chemical | `interact` |
| DDF | DDF | `affect`, `is a` |
| DDF | Human / Animal | `target` |
| Drug | Chemical / Drug | `interact` |
| Drug | DDF | `change effect` |
| Human / Animal / Microbiome | Biomedical Technique | `used by` |
| Microbiome | Anatomical Location / Human / Animal | `located in` |
| Microbiome | Gene | `change expression` |
| Microbiome | DDF | `is linked to` |
| Microbiome | Microbiome | `compared to` |

---

## Dataset Statistics

### Collection Overview

| Collection | Documents | Total Entities | Avg. Entities / Doc | Total Relations | Avg. Relations / Doc |
|---|---:|---:|---:|---:|---:|
| Train Gold | 639 | 20,530 | 32.13 | 8,556 | 13.39 |
| Train Silver | 811 | 26,134 | 32.22 | 10,907 | 13.45 |
| Train Silver 2025 | 499 | 15,275 | 30.61 | 10,616 | 21.27 |
| Train Bronze | 2,972 | 89,987 | 30.28 | 29,692 | 9.99 |
| Development Set | 80 | 2,521 | 31.51 | 1,261 | 15.76 |

---

## Evaluation

All subtasks are evaluated using standard Information Extraction metrics:

- macro-average precision
- macro-average recall
- macro-average F1-score
- micro-average precision
- micro-average recall
- micro-average F1-score

For all subtasks, the official leaderboard reference metric is:

```text
Micro-average F1-score
```

Micro-F1 is used as the main ranking metric because it better accounts for class imbalance.

### Definitions

Let:

- `TP` = true positives
- `FP` = false positives
- `FN` = false negatives
- `L` = set of labels

For subtasks 6.1.1 and 6.1.2, `L` refers to entity labels.

For subtasks 6.2.1 and 6.2.2, `L` refers to triples of:

```text
(subject label, predicate, object label)
```

### Macro Precision

```text
P_macro = average over labels of TP_l / (TP_l + FP_l)
```

### Macro Recall

```text
R_macro = average over labels of TP_l / (TP_l + FN_l)
```

### Macro F1

```text
F1_macro = 2 * (P_macro * R_macro) / (P_macro + R_macro)
```

### Micro Precision

```text
P_micro = sum(TP_l) / (sum(TP_l) + sum(FP_l))
```

### Micro Recall

```text
R_micro = sum(TP_l) / (sum(TP_l) + sum(FN_l))
```

### Micro F1

```text
F1_micro = 2 * (P_micro * R_micro) / (P_micro + R_micro)
```

---

## Baseline Results

Official baseline scores on the development set:

| Subtask | Macro-P | Macro-R | Macro-F1 | Micro-P | Micro-R | Micro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| NER | 0.7114 | 0.7480 | 0.7267 | 0.7782 | 0.8221 | 0.7996 |
| NERD | 0.3820 | 0.4045 | 0.3916 | 0.4281 | 0.4522 | 0.4398 |
| Mention-level RE | 0.3660 | 0.2862 | 0.3003 | 0.4462 | 0.3453 | 0.3893 |
| Concept-level RE | 0.1009 | 0.1021 | 0.0966 | 0.1409 | 0.1292 | 0.1348 |

---

## Submission Requirements

Participants may submit runs for any or all of the four subtasks independently.

Each group can submit a maximum of **25 runs per subtask**.

All runs must be submitted in a single zip archive named:

```text
teamID_GutBrainIE_2026.zip
```

Inside the archive, each run must be placed in a separate folder named:

```text
teamID_taskID_runID_systemDesc
```

Where:

| Component | Description |
|---|---|
| `teamID` | Team identifier chosen during CLEF 2026 registration. |
| `taskID` | Task identifier. Must be one of `T611`, `T612`, `T621`, `T622`. |
| `runID` | Run identifier containing only letters and numbers. |
| `systemDesc` | Optional short system description. No spaces or special characters. |

Task IDs:

| Subtask | Task ID |
|---|---|
| Subtask 6.1.1 NER | `T611` |
| Subtask 6.1.2 NERD | `T612` |
| Subtask 6.2.1 Mention-level RE | `T621` |
| Subtask 6.2.2 Concept-level RE | `T622` |

Each run folder must contain exactly two files:

```text
teamID_taskID_runID_systemDesc.json
teamID_taskID_runID_systemDesc.meta
```

### Metadata File Requirements

The `.meta` file should briefly describe the approach and include:

- Team ID
- Task ID
- Run ID
- Type of training applied
- Pre-processing methods
- Training data used
- Relevant details of the run
- Link to a GitHub repository enabling reproducibility

Submissions that do not follow the required naming and folder structure may be rejected.

---

## Example Submission Formats

Submissions must follow the JSON format used in the datasets and include only the field relevant to the selected subtask.

### NER: Subtask 6.1.1

Use the `entities` field **without** the `uri` subfield.

```json
{
  "34870091": {
    "entities": [
      {
        "start_idx": 75,
        "end_idx": 82,
        "location": "title",
        "text_span": "patients",
        "label": "human"
      },
      {
        "start_idx": 250,
        "end_idx": 270,
        "location": "abstract",
        "text_span": "intestinal microbiome",
        "label": "microbiome"
      }
    ]
  }
}
```

### NERD: Subtask 6.1.2

Use the `entities` field **with** the `uri` subfield.

```json
{
  "34870091": {
    "entities": [
      {
        "start_idx": 75,
        "end_idx": 82,
        "location": "title",
        "text_span": "patients",
        "label": "human",
        "uri": "http://id.nlm.nih.gov/mesh/D010361"
      },
      {
        "start_idx": 250,
        "end_idx": 270,
        "location": "abstract",
        "text_span": "intestinal microbiome",
        "label": "microbiome",
        "uri": "http://purl.obolibrary.org/obo/NCIT_C93019"
      }
    ]
  }
}
```

### Mention-Level RE: Subtask 6.2.1

Use the `mention_level_relations` field.

```json
{
  "34870091": {
    "mention_level_relations": [
      {
        "subject_text_span": "intestinal microbiome",
        "subject_label": "microbiome",
        "predicate": "located in",
        "object_text_span": "patients",
        "object_label": "human"
      }
    ]
  }
}
```

### Concept-Level RE: Subtask 6.2.2

Use the `concept_level_relations` field.

```json
{
  "34870091": {
    "concept_level_relations": [
      {
        "subject_uri": "http://purl.obolibrary.org/obo/NCIT_C93019",
        "subject_label": "microbiome",
        "predicate": "located in",
        "object_uri": "http://id.nlm.nih.gov/mesh/D010361",
        "object_label": "human"
      }
    ]
  }
}
```

---

## Recommended Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── train/
│   ├── dev/
│   └── test/
├── notebooks/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_ner.py
│   ├── train_nerd.py
│   ├── train_re.py
│   ├── predict.py
│   ├── evaluate.py
│   └── submission.py
├── runs/
│   └── README.md
├── submissions/
│   └── README.md
└── scripts/
    ├── validate_submission.sh
    └── create_submission_zip.sh
```

---

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
transformers
datasets
scikit-learn
pandas
numpy
tqdm
seqeval
spacy
```

Adjust dependencies based on the selected approach.

---

## Running the Project

### 1. Prepare Data

Place the official challenge data under the `data/` directory after registration and download.

Example:

```text
data/
├── train/
│   ├── gold.json
│   ├── silver.json
│   ├── silver_2025.json
│   └── bronze.json
├── dev/
│   └── dev.json
└── test/
    └── test.json
```

Do not commit the official dataset unless the license explicitly permits redistribution.

### 2. Train a Model

Example command:

```bash
python src/train_ner.py \
  --train data/train/gold.json \
  --dev data/dev/dev.json \
  --output runs/ner_baseline
```

### 3. Generate Predictions

Example command:

```bash
python src/predict.py \
  --task T611 \
  --model runs/ner_baseline \
  --input data/test/test.json \
  --output submissions/teamID_T611_run1_baseline.json
```

### 4. Validate Submission Format

Use the official validation script when available:

```bash
python validate_submission.py submissions/teamID_T611_run1_baseline.json
```

### 5. Create Submission Archive

Expected archive name:

```text
teamID_GutBrainIE_2026.zip
```

Example folder layout before zipping:

```text
teamID_GutBrainIE_2026/
└── teamID_T611_run1_baseline/
    ├── teamID_T611_run1_baseline.json
    └── teamID_T611_run1_baseline.meta
```

Create the zip:

```bash
zip -r teamID_GutBrainIE_2026.zip teamID_GutBrainIE_2026/
```

---

## Reproducibility Checklist

Before submitting, verify that:

- [ ] The team ID is consistent across all files and folders.
- [ ] The task ID is one of `T611`, `T612`, `T621`, `T622`.
- [ ] The run ID contains only letters and numbers.
- [ ] Folder and file names contain no spaces or special characters.
- [ ] Each run folder contains one `.json` and one `.meta` file.
- [ ] The JSON file contains only the field required for the selected subtask.
- [ ] NER submissions do not include `uri`.
- [ ] NERD submissions include `uri`.
- [ ] Mention-level RE submissions use `mention_level_relations`.
- [ ] Concept-level RE submissions use `concept_level_relations`.
- [ ] The official validation script passes.
- [ ] The `.meta` file includes training data, preprocessing, training method, and GitHub repository link.
- [ ] The final zip file is named `teamID_GutBrainIE_2026.zip`.

---

## Important Dates

| Event | Date |
|---|---|
| Training data release | Available upon registration |
| Registration closes | April 23, 2026 |
| Test data release | April 28, 2026 |
| Runs submission deadline | May 7, 2026 |
| Evaluation results released | May 19, 2026 |
| Participant and position paper deadline | May 28, 2026 |
| Paper acceptance notification | June 30, 2026 |
| Camera-ready participant papers | July 6, 2026 |
| GutBrainIE CLEF Workshop | September 21-24, 2026, Jena, Germany |

---

## Participant Paper

Participants are expected to write a report describing:

- the submitted system
- the selected subtasks
- training data used
- preprocessing
- model architecture or method
- features used for prediction
- experimental results
- error analysis and insights

Accepted reports will be published in the CLEF 2026 Working Notes at CEUR-WS. The expected report length is 10-20 pages, excluding references.

---

## References

- GutBrainIE @ CLEF 2026 official challenge page
- BioASQ Lab 2026
- CLEF 2026
- Official GutBrainIE baseline repository
- Official GutBrainIE annotation guidelines
- Official GutBrainIE validation and evaluation scripts

---

## License

This repository template does not include the official challenge datasets. Please follow the licensing and redistribution terms provided by the GutBrainIE organizers.
