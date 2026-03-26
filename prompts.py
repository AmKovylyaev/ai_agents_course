"""LLM prompt templates for each pipeline step."""

STEP1_EDA_PROMPT = """You are an ML engineer. Write Python code to perform Exploratory Data Analysis.

Context:
- Train data path: {train_path}
- Test data path: {test_path}
- Session directory: {session_dir}
- Use only 20% of training data for speed (sample with random_state=42)

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
- Session directory: {session_dir}
- Previous error (if retry): {last_error}
- Previous code (if retry): {previous_code}

Requirements:
1. Load train data (20% subset with random_state=42)
2. Identify target column (usually 'target', 'label', or last column)
3. Prepare features (handle missing values, select numeric columns)
4. Split into train/validation (80/20, random_state=42)
5. Train a model (start simple: RandomForestClassifier with n_estimators=50, random_state=42)
6. Save model to session_dir/models/model.joblib
7. Print training metrics
8. Update state dict with: target_column, model_path, X_train_shape, X_val_shape

Output ONLY executable Python code in a ```python code block.
Use sklearn, joblib, pandas, numpy.
"""

STEP3_EVAL_PROMPT = """You are an ML engineer. Write Python code to evaluate the trained model locally.

Context:
- Model path: {model_path}
- Session directory: {session_dir}
- Previous error (if retry): {last_error}

Requirements:
1. Load the model from model_path using joblib
2. Load train data again (20% subset) and recreate the same train/val split (80/20, random_state=42)
3. Identify target column (same as training)
4. Prepare validation features (same columns as training)
5. Make predictions on validation set
6. Calculate metrics: accuracy, F1-macro, confusion matrix
7. Save metrics to session_dir/reports/local_metrics.json
8. Print evaluation results
9. Update state dict with: local_metrics (dict)

Output ONLY executable Python code in a ```python code block.
Use sklearn, joblib, pandas, numpy.
"""

STEP4_SUBMISSION_PROMPT = """You are an ML engineer. Write Python code to create a submission file.

Context:
- Model path: {model_path}
- Test data path: {test_path}
- Sample submission path: {sample_submission_path}
- Session directory: {session_dir}
- Target column used in training: {target_column}
- Previous error (if retry): {last_error}

Requirements:
1. Load the model using joblib
2. Load test data
3. Prepare features (same numeric columns as training)
4. Make predictions
5. Load sample_submission.csv to get correct format
6. Create submission CSV EXACTLY matching sample_submission.csv format:
   - Copy sample_submission to a new dataframe
   - REPLACE the prediction column values with your predictions (DO NOT add new columns!)
   - Keep ONLY the columns from sample_submission (usually id/index and prediction column)
   - The final submission must have THE SAME columns as sample_submission
7. Save to session_dir/submission.csv
8. Update state dict with: submission_path

CRITICAL: The submission file must have EXACTLY the same columns as sample_submission.csv!
If sample_submission has columns [index, prediction], your submission must have [index, prediction] only.

Output ONLY executable Python code in a ```python code block.
Use pandas, joblib, numpy.
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
