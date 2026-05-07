# Hybrid Anime Recommender System - Project Plan

## 1. Project Goal

The goal of this course project is to build a **hybrid recommender system** for anime titles using the processed Anime dataset from:

`https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database`

The project will demonstrate most of the recommender-system concepts covered so far in the course:

- non-personalized recommendations
- content-based filtering
- collaborative filtering
- matrix factorization
- hybrid recommender architectures
- offline evaluation
- ranking metrics
- optional robustness or deep-learning extensions

The final system should be able to recommend anime for a given user and compare several recommendation strategies using objective metrics.

## 2. Dataset

We will use the **Anime** dataset because it has a strong balance of interaction data and item metadata.

Available files:

- `anime_ratings.dat`
  - explicit user ratings
  - columns: `User_ID`, `Anime_ID`, `Feedback`
  - rating range: 1-10

- `anime_history.dat`
  - implicit user interaction history
  - columns: `User_ID`, `Anime_ID`, `Feedback`
  - feedback value is 1, meaning the user interacted with the anime

- `anime_info.dat`
  - item metadata
  - columns: `anime_ids`, `name`, `genre`, `type`, `episodes`, `rating`, `members`

This dataset is suitable because it supports both **rating prediction** and **top-N recommendation**, while the genre/type metadata allows us to build content-based and hybrid models.

## 3. Main Implementation Stages

### Stage 1: Data Loading and Cleaning

We will start by loading the three `.dat` files and preparing them for modeling.

Tasks:

- read all files with tab-separated parsing
- validate column names and data types
- remove or handle missing values
- normalize numeric metadata such as `episodes`, `rating`, and `members`
- split `genre` into usable features
- check dataset sparsity, number of users, number of items, and rating distribution

Why this matters:

Clean and well-understood data is the foundation for every recommender method. This stage also gives us useful statistics for the final report.

Expected outputs:

- cleaned ratings dataframe
- cleaned implicit-history dataframe
- cleaned item metadata dataframe
- exploratory data analysis summary

## 4. Recommendation Methods

### Stage 2: Non-Personalized Baselines

We will implement simple recommendation baselines that do not depend on a specific user's taste.

Models:

- most popular anime by number of interactions
- highest average rating
- damped mean rating
- popularity-adjusted score using `members`

Why this matters:

Non-personalized recommenders are simple but important baselines. A personalized model should outperform them, or at least provide better diversity and relevance.

Course connection:

- Week 02: non-personalized recommendations, scoring, ranking, damped means

Expected outputs:

- top anime by popularity
- top anime by damped mean score
- baseline ranking metrics

### Stage 3: Content-Based Recommender

We will build item profiles from anime metadata.

Features:

- genre terms
- anime type: TV, Movie, OVA, etc.
- number of episodes
- global anime rating
- member count
- optionally title/name terms

Methods:

- encode metadata using TF-IDF and/or one-hot encoding
- compute item-item cosine similarity
- build user profiles from anime the user rated highly
- recommend anime similar to the user's profile

Why this matters:

Content-based filtering can recommend items using item attributes, even when collaborative data is sparse. It also makes recommendations more explainable because we can say they are based on genre/type similarity.

Course connection:

- Week 02: content-based filtering, TF-IDF, cosine similarity, user profiles

Expected outputs:

- similar-anime lookup
- personalized content-based recommendations
- explanation examples such as "recommended because the user likes Action and Fantasy anime"

### Stage 4: Collaborative Filtering

We will implement collaborative filtering using the explicit rating matrix.

Candidate methods:

- user-based nearest-neighbor collaborative filtering
- item-based nearest-neighbor collaborative filtering
- adjusted cosine similarity or Pearson correlation

Why this matters:

Collaborative filtering captures patterns that are not directly visible from metadata. For example, two anime may be recommended together because similar users rated both highly, even if their genres differ.

Course connection:

- Week 04: user-based CF, item-based CF, similarity measures, neighborhood selection

Expected outputs:

- rating predictions for user-anime pairs
- top-N collaborative recommendations
- comparison between user-based and item-based approaches

### Stage 5: Matrix Factorization

We will implement or use a matrix factorization model for explicit ratings.

Candidate models:

- SVD-style latent factor model
- biased matrix factorization with global mean, user bias, item bias, and latent factors

Prediction form:

```text
predicted_rating = global_mean + user_bias + item_bias + user_vector * item_vector
```

Why this matters:

Matrix factorization is one of the central model-based recommender techniques. It can handle sparse rating matrices better than direct nearest-neighbor approaches and captures hidden preference factors.

Course connection:

- Week 04: SVD, latent factors, regularization, SGD optimization

Expected outputs:

- predicted ratings
- RMSE/MAE comparison against baselines
- learned user/item latent representations

### Stage 6: Implicit-Feedback Ranking

We will use `anime_history.dat` to train or evaluate top-N recommendation behavior.

Methods:

- popularity baseline on implicit interactions
- item-item similarity from interaction history
- optional implicit matrix factorization or ranking model

Why this matters:

Many real recommender systems work with implicit feedback such as views, clicks, purchases, or watch history. This part shows that the project is not limited to explicit ratings.

Course connection:

- Week 02: implicit vs explicit feedback
- Week 05: ranking evaluation
- Week 08: session/interaction-based recommendation ideas

Expected outputs:

- top-N recommendations from implicit data
- Precision@K, Recall@K, MAP@K, and nDCG@K

### Stage 7: Hybrid Recommender

We will combine several model outputs into a final hybrid recommender.

Possible hybrid design:

```text
hybrid_score =
    alpha * matrix_factorization_score
  + beta  * content_based_score
  + gamma * popularity_score
```

Alternative hybrid strategy:

- use collaborative filtering when the user has enough ratings
- use content-based recommendations for sparse users
- use popularity/damped mean for cold-start users

Why this matters:

The hybrid model is the main project deliverable. It uses multiple sources of evidence and directly addresses weaknesses of individual approaches, especially cold start, sparsity, and overspecialization.

Course connection:

- Week 06: hybrid architectures, weighted hybrids, switching hybrids, ensembles

Expected outputs:

- final recommendation function
- tunable hybrid weights
- comparison against individual recommenders
- examples for normal users, sparse users, and cold-start-like cases

## 5. Evaluation Plan

We will evaluate the models using both rating-prediction metrics and ranking metrics.

### Rating Prediction Metrics

Used for models trained on `anime_ratings.dat`.

- MAE
- RMSE

These metrics answer:

> How close are the predicted ratings to the actual user ratings?

### Ranking Metrics

Used for top-N recommendation lists.

- Precision@K
- Recall@K
- MAP@K
- nDCG@K

These metrics answer:

> Are the most useful recommendations placed near the top of the list?

### Additional Recommender Properties

If time allows, we will also report:

- coverage: how many anime can be recommended
- diversity: how different the recommended anime are from each other
- novelty or serendipity: whether the model avoids recommending only very popular anime

Course connection:

- Week 05: offline evaluation, RMSE, MAE, Precision, Recall, MAP, nDCG, coverage, diversity, serendipity

## 6. Optional Advanced Extensions

These are not required for the core project, but they can make the final work stronger.

### Option A: Robustness Experiment

We can simulate a simple push or nuke attack by injecting fake ratings for a target anime and measuring how much the recommendation score changes.

Why this is useful:

It connects the project to recommender-system attacks and robustness.

Course connection:

- Week 07: push attacks, nuke attacks, random attacks, average attacks, stability

### Option B: Simple Neural Recommender

We can implement a small neural collaborative filtering model with user and item embeddings.

Why this is useful:

It connects the project to modern deep-learning recommenders while keeping the model manageable.

Course connection:

- Week 08: embeddings, deep collaborative filtering, NeuMF

### Option C: Reinforcement Learning Framing

We can describe a future RL version of the system where:

- state = user profile and recent interactions
- action = recommend an anime
- reward = rating, click, watch, or completion
- policy = strategy for choosing recommendations

This does not need to be fully implemented unless there is enough time.

Course connection:

- Week 09: RL, MDP, reward, policy, Q-learning

## 7. Final Deliverables

The final project should include:

- cleaned and reproducible data-loading pipeline
- exploratory data analysis
- at least one non-personalized baseline
- at least one content-based recommender
- at least one collaborative filtering model
- matrix factorization model
- hybrid recommender
- evaluation results and comparison table
- sample recommendations for selected users
- short explanation of limitations and possible improvements

## 8. Recommended Minimum Scope

If time becomes limited, the minimum strong version of the project should include:

1. data cleaning and analysis
2. popularity and damped-mean baselines
3. content-based recommender using genres and type
4. item-based collaborative filtering
5. matrix factorization
6. weighted hybrid recommender
7. RMSE/MAE plus Precision@K/nDCG@K evaluation

This version would still cover the most important course topics and produce a complete, defensible recommender-system project.

