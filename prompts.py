"""LLM prompt templates for each pipeline step."""

STEP1_EDA_PROMPT = """You are an ML engineer. Write Python code to perform Exploratory Data Analysis.

Context:
- Train data path: {train_path}
- Test data path: {test_path}
- Session directory: {session_dir}
- Use only {train_sample_pct}% of training data for speed (sample frac={train_sample_frac}, random_state=42)

Requirements:
1. Load train and test CSV files using pandas
2. Show basic statistics (shape, columns, dtypes, missing values)
3. Create visualizations if helpful (save to session_dir/reports/)
4. Save a summary report to session_dir/reports/eda_summary.txt
5. Save the loaded dataframes to JSON-serializable state dict with keys: train_shape, test_shape, columns, missing_values

Output ONLY executable Python code in a ```python code block.
Use pandas, numpy, and matplotlib/seaborn if needed.
The code should work standalone without user input.
At the end, update the 'state' dict with useful information.
"""

STEP2_TRAIN_PROMPT = """You are an ML engineer. Write Python code to train a classification model.

Context:
- Train data path: {train_path}
- Test data path: {test_path}
- Session directory: {session_dir}
- Previous error (if retry): {last_error}
- Previous code (if retry): {previous_code}

Data constraints:
- The dataset contains mixed types including text/string columns.
- You must use only numeric features for the model. Drop all non-numeric columns before training.
- Use {train_sample_pct}% of the data (sample frac={train_sample_frac}, random_state=42) for speed.
- Split into train/validation 80/20 with random_state=42.

Requirements:
1. Load and sample the training data
2. Identify the target column (usually 'target', 'label', or the last column)
3. Select only numeric feature columns, exclude the target and all text/string columns
4. Handle missing values if needed
5. Train a RandomForestClassifier (n_estimators=50, random_state=42)
6. Save the model to session_dir/models/model.joblib using joblib
7. Print training metrics
8. Update state dict with: target_column, model_path, X_train_shape, X_val_shape

Output ONLY executable Python code in a ```python code block.
Use sklearn, joblib, pandas, numpy.
"""

STEP3_EVAL_PROMPT = """You are an ML engineer. Write Python code to evaluate the trained model locally.

Context:
- Train data path: {train_path}
- Model path: {model_path}
- Session directory: {session_dir}
- Previous error (if retry): {last_error}

Data constraints:
- The dataset contains mixed types including text/string columns.
- The model was trained on numeric features only. You must select the same numeric-only features for evaluation.
- Reproduce the exact same data preparation: {train_sample_pct}% sample (frac={train_sample_frac}, random_state=42), then 80/20 split (random_state=42).

Requirements:
1. Load the model from model_path using joblib
2. Load and sample the training data (same {train_sample_pct}% subset as training)
3. Identify the target column (usually 'target', 'label', or last column)
4. Select only numeric feature columns — same approach as training
5. Recreate the train/val split to get the validation set
6. Predict on the validation set
7. Calculate metrics: accuracy, F1-macro, confusion matrix
8. Save metrics to session_dir/reports/local_metrics.json
9. Print evaluation results
10. Update state dict with: local_metrics (dict)

Output ONLY executable Python code in a ```python code block.
Use sklearn, joblib, pandas, numpy.
"""

STEP4_SUBMISSION_PROMPT = """You are an ML engineer. Write Python code to create a submission file.

Context:
- Model path: {model_path}
- Test data path: {test_path}
- Sample submission path: {sample_submission_path}
- Session directory: {session_dir}
- Previous error (if retry): {last_error}

Data constraints:
- The dataset contains mixed types including text/string columns.
- The model was trained on numeric features only. Use the same numeric-only features for prediction.

Requirements:
1. Load the model using joblib
2. Load the test data and select only numeric feature columns
3. Make predictions on the test set
4. Load sample_submission.csv — it defines the exact output format
5. Create the submission by copying the sample and replacing its last column with your predictions
6. The output must contain EXACTLY the same columns as sample_submission.csv — do not add or rename columns
7. Save to session_dir/submission.csv
8. Update state dict with: submission_path

Output ONLY executable Python code in a ```python code block.
Use pandas, joblib, numpy.
"""

STEP_VERIFY_PROMPT = """You are a Senior Data Scientist (Judge). Your task is to evaluate the results of a Machine Learning experiment for a Kaggle competition: predicting occupancy days (or idle days) for real estate rentals. This is a regression task; the evaluation metric is MSE (Mean Squared Error).

Context:
- EDA Summary: {eda_summary}
- Local Metrics (validation MSE and possibly other regression metrics): {local_metrics}
- Previous Code: {previous_code}

Requirements:
1. Analyze if the MSE is reasonable for this regression task. Compare with a simple baseline (e.g., predicting the mean target value). If the MSE is high relative to the target variance, it indicates poor performance.
2. Check for overfitting if possible (e.g., by looking at training vs validation error, though we only have validation).
3. Determine if the results are "SUFFICIENT" to proceed to submission or "NEED_REFINEMENT".
4. If refinement is needed, provide concrete, actionable suggestions. Focus on:
   - Feature engineering (e.g., date/time features, property characteristics, lag variables)
   - Model choice (e.g., XGBoost, LightGBM, neural networks, ensemble)
   - Hyperparameter tuning
   - Handling outliers or target transformation (log, sqrt)
   - Any other data-specific insights from EDA

Your output MUST be a JSON-formatted string with:
- "decision": "SUFFICIENT" or "NEED_REFINEMENT"
- "reasoning": "Short explanation of your choice, referencing MSE value and potential baseline."
- "suggestions": "What to improve if REFINEMENT is needed (e.g., 'try extracting day-of-week features from timestamp', 'use LightGBM with hyperparameter tuning', 'apply log transform to target')"

Example: {{"decision": "NEED_REFINEMENT", "reasoning": "Validation MSE is 125.4, while baseline mean prediction MSE is 200, so it's better but still high.", "suggestions": "Add date features (month, day of week), use XGBoost with early stopping, and try log transformation on target."}}
"""

STEP7_REPORT_PROMPT = """You are an ML engineer. Write Python code to generate a final report.

Context:
- Session directory: {session_dir}
- EDA insights available in: session_dir/reports/eda_summary.txt
- Local metrics available in: session_dir/reports/local_metrics.json
- Submission status: {submit_ok}
- Kaggle scores: public={public_score}, private={private_score}

Requirements:
1. Load EDA summary from file if exists
2. Load local metrics from file if exists
3. Compile all results into a structured report
4. Save as JSON: session_dir/reports/final_report.json
5. Save as readable text: session_dir/reports/final_report.txt
6. Include in the report:
   - EDA insights
   - Model parameters
   - Local validation metrics
   - Kaggle submission results
   - Lessons learned and next steps

Output ONLY executable Python code in a ```python code block.
Use json, pathlib.
"""
