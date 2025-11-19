# Part A additional questions

## 1. What sort of testing would you have on the training code?

For production ML pipelines, I would d implement testing at multiple levels:

**Unit tests** as the foundation: 
- Each preprocessing function in `preprocessing.py` can be covered by isolated tests. 
- For instance, testing that the `pdays` transformation correctly creates the `was_contacted_before` binary feature and handles the case of never-contacted customers (-1 value). 
- Similarly, the outlier removal logic could be tested to verify that the calculations work correctly and that threshold removals for `previous` and `days_since_contact` are as expected. 
- Each feature engineering step should have assertions checking that output shapes, data types, and value ranges match expectations.

**Data validation tests** could be used for catching issues before training begins: 
- I would add tests to validate schema consistency, check for expected value ranges (e.g., age between 18-100), verify categorical values match expected sets, and possibly identify data drift. 
- Tests should verify that the target variable only contains yes/no values, that no unexpected nulls appear after preprocessing, and that feature names remain consistent across runs, etc. 

**Integration tests** to verify the complete pipeline orchestration in `pipeline.py`: 
- A pipeline test that runs train.py using a synthetic dataset. This ensures the pipeline runs assuming correct data and that it succeeds end-to-end from data load to packaging without issues. 
- Possible tests could include: 
  - Checks that preprocessing outputs have the expected shape and feature names
  - Trained model can actually generate predictions on sample data
  - Serialised artefacts can be loaded and used correctly. 

**Model performance \ regression tests** to ensure that new model is not significantly worse than the baseline performance. 
- E.g. after training, automatically check that metrics like F1 score exceed minimum thresholds (already implemented with the acceptance threshold in config in this implementation). 
- We can store baseline metrics and alert if new models underperforms significantly.  

Practically speaking, I'd run these tests in a CI/CD pipeline using Buildkite or GitHub Actions with: 
- Unit tests on every commit or on pull requests
- Integration tests on pull requests
- Heavier regression tests on major builds

## 2. As you're conducting experimentation to improve the model, how do you make sure the necessary configs, data, hyperparameters and evaluation outcomes are stored?

**MLflow and code versioning**: 
In the current implementation, I used config-driven approach and saving artefacts in the model folder, as well as using repository for tracking changes and for config versioning. 
To properly track experiments and artefacts, I would suggest to use MLflow or a similar tool.  

Some suggestions:  
- The `config.yaml` file should be version-controlled in Git alongside the code.
- I would further extend the config to include an `experiment_name` field to organise related runs. 
- Overall, I would further improve the training pipeline by changing it to to log everything to MLflow. 
  - Logging all hyperparameters from `config.yaml` using `mlflow.log_params()`
  - Recording metrics like F1 score, precision, recall at each evaluation step with `mlflow.log_metrics()`
  - Storing the trained model and its metadata as an MLflow model artefact using `mlflow.xgboost.log_model()` (instead of creating picke files and json files)
  - Saving the fitted preprocessor as an artefact with `mlflow.log_artifact()`
  - Logging feature importance plots and confusion matrices for visual comparison. 

MLflow is great for this as it provides queryable history and comparison tools. 

**Data versioning**: I would use DVC (Data Version Control) to track dataset versions, or AWS S3 with object versioning enabled. Each MLflow run should log the data version or S3 URI used for training using `mlflow.log_param("data_version", "v2024-11-15")`. This ensures complete reproducibility.  

**Artefact registry**: Rather than scattering pickle files in directories, use MLflow's model registry to promote models through stages from local to dev and to production). 
- The `package_model.py` output would be registered in MLflow with metadata like feature names, preprocessing logic, and evaluation metrics. 
- Models can be retrieved by name and stage: `model = mlflow.pyfunc.load_model("models:/marketing_model/Production")`. Model artefacts would be stored in S3 with MLflow tracking metadata in RDS, providing a queryable catalogue of all trained models (in case of AWS implemenation). 

**Automated tracking in CI/CD**: Every training run triggered by a code change should automatically log to MLflow, with the Git SHA tagged. This creates a trail showing which code version produced which model.

## 3. You notice that your training results are different each time and fluctuate without you changing anything. What's wrong? How do you fix it?

Fluctuating results despite consistent code could be caused by various sources of randomness in the pipeline. This would require some investigation, and I would try a few things to fix that. 

**Random seed propagation**: Algorithms like XGBoost use randomness to build the model. Additionally, if the train/test splits relies on a random shuffle without a fixed state, the evaluation set changes every run. Note that I have handled this already in the implementation by defining random seed and passing it to XGBoost and to numpy (see `train.py` and `preprocessing.py`)

**Possible data ordering issues**: If the input data isn't consistently ordered, train-test splits might vary. The fix is to explicitly sort the DataFrame by specific keys before any splitting (a possible improvement, I did not implement this in code).

**Parallel processing non-determinism**: XGBoost's `n_jobs: -1` setting enables multi-threading, which can introduce non-determinism in how tree splits are evaluated. The effect is usually quite minor, though. 

**Library version differences**: Different versions of XGBoost, scikit-learn, or pandas may produce different results due to algorithm improvements, bug fixes, or numerical precision changes. The solution is to pin all dependencies precisely in `requirements.txt` using exact versions, which I would normally suggest for a production grade solution. Overall I'd suggest use Docker images with pinned dependencies rather than installing packages dynamically. 

**Debugging**: To isolate the issues, I would:
1. Log the first few rows of data after each preprocessing step to verify consistency
2. Log feature statistics (mean, std) to detect numerical drift
3. Save and compare train/test splits across runs and compare 
4. Try training with `n_jobs=1` and `random_state` set everywhere.
5. Ensure the library versions are consistent.

## 4. How do you approach setting up a retraining strategy?

Ideally, I would implement monitoring for both data drift and model performance degradation. There could also be a process for scheduled retraining.  

**Data drift detection** would compare incoming data distributions against the baseline (training) set. For this banking dataset:
- Statistical tests: KS test for continuous features (age, balance, duration), chi-squared test for categorical features (job, education, marital status)
- Population stability index (PSI) across all features, with thresholds like PSI > 0.1 (minor shift), PSI > 0.2 (moderate shift requiring investigation)
- Monitor feature correlations - do relationships between features change (e.g., age vs balance correlation shifts)? 
- Track class distribution - does the proportion of positive cases (term deposit subscribers) change significantly from the training baseline? 

**Model performance drift** to track prediction quality over time:
- If ground truth labels becomes available alter (customers eventually subscribe or not), calculate scores on recent predictions
- Monitor prediction distributions - if the model suddenly predicts "yes" much more or less frequently, something has changed
- Track confidence scores - if average prediction probabilities drift towards 0.5 (low confidence), the model is becoming uncertain
- Business metrics - if conversion rates or campaign ROI drop, the model may be missing new patterns

**Scheduled retraining (automated pipeline)**: 
- For this dataset and business problem I would suggest to set monthly or quarterly retraining as customer behaviour and economic conditions would likely change gradually rather than abruptly
- Retraining can be done on a sliding window of recent data (e.g., last 12-18 months) to stay relevant 
- I would suggest adding promotion gates for results to be reviewed & approved before moving the model to production
- Alternatively, can run A/B test or shadow deployment (model receives traffic but results aren't used) for 1-2 weeks to validate the results before moving to the new model

**Rollback strategy**: If a new model causes issues in production, MLflow makes rollback easy as you can transition the previous model back to production stage. 

The key principle I would suggest is **automated monitoring with human-in-the-loop for key deployment decisions**.  
