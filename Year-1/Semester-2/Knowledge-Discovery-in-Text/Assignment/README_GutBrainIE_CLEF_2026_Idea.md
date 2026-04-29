For a **course project**, I would recommend implementing:

## Best choice: **Subtask 6.1.1 — Named Entity Recognition / NER**

This is the most appropriate and realistic task.

You need to detect biomedical entity mentions in PubMed titles/abstracts and classify them into labels such as:

```text
bacteria, chemical, DDF, food, drug, gene, microbiome, human, animal, etc.
```

Why this is good for a course assignment:

* It is a clear NLP sequence-labeling task.
* The input and output are understandable.
* You can implement several levels of complexity:

  * rule-based baseline,
  * classical ML baseline,
  * transformer-based model.
* Evaluation is straightforward: precision, recall, F1.
* It is easier to explain in a report and presentation.
* You do not need external biomedical concept linking resources.

A good project title could be:

> **Biomedical Named Entity Recognition for Gut–Brain Axis Literature**

Recommended implementation:

```text
Input: PubMed title + abstract
Output: entity spans + entity labels
Model: BioBERT / PubMedBERT / SciBERT / BiLSTM-CRF / baseline rules
Evaluation: micro-F1 and macro-F1
```

---

## Optional extension: **Subtask 6.2.1 — Mention-Level Relation Extraction**

This could be added as a second phase **only after NER works**.

Here you detect relations between entity mentions, for example:

```text
bacteria → influence → disease
chemical → impact → microbiome
microbiome → is linked to → DDF
```

Why it is useful:

* It makes the project more complete.
* It shows information extraction beyond just entity detection.
* You can reuse the entities from NER.

But it is more difficult because you need to predict relations between pairs of entities, not only label words.

A realistic course-project version would be:

```text
Phase 1: Detect entities with NER
Phase 2: Generate candidate entity pairs
Phase 3: Classify relation type or no-relation
```


## My recommendation

The best scope would be:

```text
Main task:
Subtask 6.1.1 — Named Entity Recognition

Optional extension:
Subtask 6.2.1 — Mention-Level Relation Extraction
```

For the course assignment, I would frame it like this:

> We will implement a biomedical information extraction pipeline for GutBrainIE. The main focus will be Named Entity Recognition over PubMed abstracts. If time allows, we will extend the system with mention-level relation extraction between detected biomedical entities.
