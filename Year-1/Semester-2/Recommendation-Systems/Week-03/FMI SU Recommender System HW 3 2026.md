Description
This kaggle competition is part of the "Recommender Systems" course at Sofia University, FMI.

The domain of the competition is books recommedation as you have data about:

Users - id, location, age
Books - id, title, author, year, publisher
and you have ratings from 0 to 10 which user id gave to book id

Users and Books files are the same for training and testing as
for training you have 862,335 ratings which the users gave to the books and for testing you have 287,445 ratings which you need to predict.

The expectation is that you will try content based, collaborative filtering and/or hybrid approach as the evaluation will be with RMSE.

Evaluation
The evaluation metric for this competition is RMSE ## Submission Format **For every userid,bookid in the test_dataset**, submission files should contain 3 columns: `Id`,`Rating`. The ratings should be integer. The file should contain a header and have the following format: ``` `Id`,`Rating` 0,9 1,10 2,5 3,10 4,0 ```

Citation
MilenChechev. FMI SU Recommender System HW 3/2026. https://kaggle.com/competitions/fmi-su-recommender-system-hw-3-2026, 2026. Kaggle.
