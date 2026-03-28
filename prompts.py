"""LLM prompt templates for each pipeline step."""

# ---------------------------------------------------------------------------
# Step 1 – EDA
# ---------------------------------------------------------------------------

STEP1_PLANNER_PROMPT = """You are an ML engineering planner. Create a concise plan for Exploratory Data Analysis.

Context:
- Train data path: {train_path}
- Test data path: {test_path}
- Session directory: {session_dir}
- Use only {train_sample_pct}% of training data (sample frac={train_sample_frac}, random_state=42)
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}

Output a numbered step-by-step plan covering:
1. How to load and sample the data
2. Which statistics to compute for ALL columns (shape, dtypes, missing values, unique counts)
3. How to profile numeric columns (min, max, mean, std, correlations)
4. How to profile categorical/text columns (cardinality, top values)
5. Target variable analysis (distribution, class balance)
6. Which visualizations to create (if any)
7. Where to save outputs (SESSION_DIR/reports/)
8. Which keys to add to the state dict (include column type lists: numeric_columns, categorical_columns)

If there was a previous error or verifier feedback, adjust the plan to avoid the same mistake.
Output ONLY the plan as a numbered list.
"""

STEP1_EDA_PROMPT = """You are an ML engineer. Write Python code that follows the plan below.

Plan:
{plan}

Paths:
- Train data: {train_path}
- Test data: {test_path}
- Session directory: {session_dir}
- Sampling: {train_sample_pct}% of training data (frac={train_sample_frac}, random_state=42)

Previous attempt (if retry):
- Error: {last_error}
- Code: {previous_code}
- Verifier feedback: {verifier_feedback}

The state dict MUST be updated with at least these keys:
  train_shape, test_shape, columns, numeric_columns, categorical_columns,
  target_column, missing_values, n_classes

Save the EDA report to session_dir/reports/eda_summary.txt.
Output ONLY executable Python code in a ```python code block.
"""

STEP1_VERIFIER_PROMPT = """You are a code verification expert. Analyze the failed EDA code and its error.

Code that failed:
{previous_code}

Error message:
{last_error}

Plan that was followed:
{plan}

Context:
- Train data path: {train_path}
- Test data path: {test_path}
- Session directory: {session_dir}

Provide specific, actionable feedback:
1. Root cause of the error
2. Exact fix (which lines to change and how)
3. Any edge cases to handle (missing files, empty data, wrong dtypes)

Output ONLY the feedback as a concise numbered list.
"""

# ---------------------------------------------------------------------------
# Step 2 – Train
# ---------------------------------------------------------------------------

STEP2_PLANNER_PROMPT = """You are an ML engineering planner. Create a concise plan for training a classification model.

EDA results from step 1 (available in state):
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}
- Target column: {target_column}
- Number of classes: {n_classes}
- Train shape: {train_shape}

Context:
- Train data is at TRAIN_DATA_PATH
- Outputs go to SESSION_DIR
- Use TRAIN_SAMPLE_PCT% of the data (frac=TRAIN_SAMPLE_FRAC, random_state=42)
- Split 80/20 with random_state=42
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}

IMPORTANT — Use ALL available columns with appropriate encoding:
- FIRST: drop the target column to get X; derive column lists from X (not from the full DataFrame).
- Build a sklearn Pipeline with ColumnTransformer. Split categoricals by cardinality:
  * Numeric columns: SimpleImputer(strategy="median") → StandardScaler
  * Low-cardinality categoricals (≤50 unique): SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore", sparse_output=False)
  * High-cardinality categoricals (>50 unique): SimpleImputer(strategy="constant", fill_value="__missing__") → OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1). OrdinalEncoder is simpler than FeatureHasher and works well with tree-based models.
- Choose a model that trains fast (must finish in under 2 minutes):
  * RandomForestClassifier(n_estimators=100, random_state=42) — fast, solid default
  * GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42) — only if few total features
  * Do NOT use default GradientBoosting params on large feature spaces
- Save the ENTIRE Pipeline object (preprocessing + model) to SESSION_DIR/models/pipeline.joblib

Output a numbered step-by-step plan covering:
1. Data loading and sampling
2. Target column identification; drop target from X
3. Splitting numeric vs categorical feature columns (from X, NOT from the full DataFrame)
4. Splitting categoricals into low-cardinality (≤50 unique) and high-cardinality (>50)
5. Building the ColumnTransformer: numeric, low-card OneHotEncoder, high-card OrdinalEncoder
6. Model selection and hyperparameters (justify your choice; must train in <2 min)
7. Assembling the full Pipeline (preprocessor + model)
8. Train/val split, fitting, and metric reporting
9. Saving the pipeline to SESSION_DIR/models/pipeline.joblib
10. State dict keys to update

If there was a previous error or verifier feedback, adjust the plan accordingly.
Output ONLY the plan as a numbered list.
"""

STEP2_TRAIN_PROMPT = """You are an ML engineer. Write Python code that follows the plan below.

Plan:
{plan}

Paths:
- Train data: {train_path}
- Session directory: {session_dir}
- Sampling: {train_sample_pct}% of data (frac={train_sample_frac}, random_state=42)

EDA results:
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}
- Target column: {target_column}
- Number of classes: {n_classes}

Previous attempt (if retry):
- Error: {last_error}
- Code: {previous_code}
- Verifier feedback: {verifier_feedback}

Hard constraints:
- FIRST drop the target column from the DataFrame to get X, then derive numeric_cols and categorical_cols from X (do NOT trust the EDA lists blindly — they may include the target). Filter: numeric_cols = [c for c in numeric_cols if c in X.columns]; same for categorical_cols.
- Build a sklearn Pipeline (ColumnTransformer + classifier) — the ENTIRE pipeline must be saved as a single joblib file to session_dir/models/pipeline.joblib.
- Fit on raw DataFrame columns (do NOT pre-transform before passing to pipeline).
- Print validation metrics (accuracy, F1-macro).
- Update state dict with: target_column, model_path, X_train_shape, X_val_shape, model_type

Output ONLY executable Python code in a ```python code block.
"""

STEP2_VERIFIER_PROMPT = """You are a code verification expert. Analyze the failed model training code and its error.

Code that failed:
{previous_code}

Error message:
{last_error}

Plan that was followed:
{plan}

Context:
- Train data path: {train_path}
- Session directory: {session_dir}

Provide specific, actionable feedback:
1. Root cause of the error
2. Exact fix (which lines to change and how)
3. Common pitfalls:
   - Not using ColumnTransformer / Pipeline correctly
   - Passing pre-transformed data instead of raw columns to the pipeline
   - Wrong column lists (numeric vs categorical)
   - Including the target column in numeric_cols or categorical_cols — it must be dropped FIRST
   - sparse_output not set to False in OneHotEncoder
   - Not saving the full pipeline (only saving the model)
   - Using FeatureHasher incorrectly — prefer OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) for high-cardinality categoricals
4. Verify that the ENTIRE pipeline is saved to pipeline.joblib (not just the classifier)

Output ONLY the feedback as a concise numbered list.
"""

# ---------------------------------------------------------------------------
# Step 3 – Eval
# ---------------------------------------------------------------------------

STEP3_PLANNER_PROMPT = """You are an ML engineering planner. Create a concise plan for local model evaluation.

Context:
- Train data is at TRAIN_DATA_PATH
- Saved pipeline (preprocessing + model) is at MODEL_PATH
- Outputs go to SESSION_DIR
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}

Key point:
- The saved pipeline.joblib contains the FULL sklearn Pipeline (ColumnTransformer + classifier).
- You do NOT need to preprocess the data manually — just pass raw DataFrame columns to pipeline.predict().
- Reproduce the exact data split: TRAIN_SAMPLE_PCT% sample (frac=TRAIN_SAMPLE_FRAC, random_state=42), 80/20 split (random_state=42).

Output a numbered step-by-step plan covering:
1. Load the pipeline from MODEL_PATH
2. Load and sample the training data (same split as training)
3. Identify target column, separate X and y
4. Recreate the 80/20 train/val split to get the validation set
5. Call pipeline.predict(X_val) — NO manual preprocessing
6. Compute metrics: accuracy, F1-macro, classification_report, confusion matrix
7. Save metrics to SESSION_DIR/reports/local_metrics.json
8. Print all metrics clearly
9. Update state dict with local_metrics

If there was a previous error or verifier feedback, adjust the plan accordingly.
Output ONLY the plan as a numbered list.
"""

STEP3_EVAL_PROMPT = """You are an ML engineer. Write Python code that follows the plan below.

Plan:
{plan}

Paths:
- Train data: {train_path}
- Pipeline (preprocessing + model): {model_path}
- Session directory: {session_dir}
- Target column: {target_column}
- Sampling: {train_sample_pct}% (frac={train_sample_frac}, random_state=42), 80/20 split (random_state=42)

Previous attempt (if retry):
- Error: {last_error}
- Code: {previous_code}
- Verifier feedback: {verifier_feedback}

Hard constraints:
- The saved pipeline.joblib is a FULL sklearn Pipeline — pass RAW DataFrame columns to pipeline.predict(), do NOT manually preprocess.
- Reproduce the exact same data split as training.
- Save metrics to session_dir/reports/local_metrics.json.
- Print all evaluation results clearly (accuracy, F1-macro, classification_report, confusion matrix).
- Update state dict with: local_metrics (dict with accuracy, f1_macro)

Output ONLY executable Python code in a ```python code block.
"""

STEP3_VERIFIER_PROMPT = """You are a code verification expert. Analyze the failed evaluation code and its error.

Code that failed:
{previous_code}

Error message:
{last_error}

Plan that was followed:
{plan}

Context:
- Train data path: {train_path}
- Model path (pipeline): {model_path}
- Session directory: {session_dir}

Provide specific, actionable feedback:
1. Root cause of the error
2. Exact fix (which lines to change and how)
3. Common pitfalls:
   - Manually preprocessing data instead of passing raw columns to pipeline.predict()
   - Wrong data split reproduction
   - Using select_dtypes or encoding — the pipeline already handles this
   - Dropping columns that the pipeline expects

Output ONLY the feedback as a concise numbered list.
"""

# ---------------------------------------------------------------------------
# Step 4 – Submission
# ---------------------------------------------------------------------------

STEP4_PLANNER_PROMPT = """You are an ML engineering planner. Create a concise plan for creating a Kaggle submission file.

Context:
- Saved pipeline (preprocessing + model) is at MODEL_PATH
- Test data is at TEST_DATA_PATH
- Sample submission is at SAMPLE_SUBMISSION_PATH (defines the EXACT output format — same columns, same order, same number of rows)
- Target column name: {target_column}
- Outputs go to SESSION_DIR
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}

Key points:
- The saved pipeline.joblib contains the FULL sklearn Pipeline (ColumnTransformer + classifier).
- Pass raw test DataFrame feature columns to pipeline.predict() — drop the target column from test data if present.
- The output submission MUST have EXACTLY the same columns as SAMPLE_SUBMISSION_PATH — no extra columns, no missing columns.

Output a numbered step-by-step plan covering:
1. Load SAMPLE_SUBMISSION_PATH first — read its column names and row count
2. Load the pipeline from MODEL_PATH
3. Load test data from TEST_DATA_PATH
4. Drop the target column from test data if it exists
5. Call pipeline.predict(X_test) — NO manual preprocessing
6. Build output DataFrame with ONLY the columns from the sample submission
7. Replace the prediction column (last column of sample) with your predictions
8. Verify output has same columns and row count as sample — assert this
9. Save to SESSION_DIR/submission.csv (index=False)
10. Update state dict

If there was a previous error or verifier feedback, adjust the plan accordingly.
Output ONLY the plan as a numbered list.
"""

STEP4_SUBMISSION_PROMPT = """You are an ML engineer. Write Python code that follows the plan below.

Plan:
{plan}

Paths:
- Pipeline (preprocessing + model): {model_path}
- Test data: {test_path}
- Sample submission: {sample_submission_path}
- Session directory: {session_dir}
- Target column: {target_column}

Previous attempt (if retry):
- Error: {last_error}
- Code: {previous_code}
- Verifier feedback: {verifier_feedback}

Hard constraints:
- The saved pipeline.joblib is a FULL sklearn Pipeline — pass RAW DataFrame columns to pipeline.predict(), do NOT manually preprocess.
- Drop the target column ("{target_column}") from test data before predicting — use errors="ignore" since test data may not have this column.
- Load sample_submission.csv FIRST to get its exact column names.
- The output CSV MUST contain EXACTLY the same columns as sample_submission.csv — no extra columns, no missing columns. Use: out_df = sample[sample.columns].copy(), then replace the last column with predictions.
- Assert that output has the same number of columns and rows as the sample before saving.
- Save to session_dir/submission.csv with index=False.
- Update state dict with: submission_path

Output ONLY executable Python code in a ```python code block.
"""

STEP4_VERIFIER_PROMPT = """You are a code verification expert. Analyze the failed submission code and its error.

Code that failed:
{previous_code}

Error message:
{last_error}

Plan that was followed:
{plan}

Context:
- Model path (pipeline): {model_path}
- Test data path: {test_path}
- Sample submission path: {sample_submission_path}
- Session directory: {session_dir}

Provide specific, actionable feedback:
1. Root cause of the error
2. Exact fix (which lines to change and how)
3. Common pitfalls:
   - Manually preprocessing instead of using pipeline.predict()
   - Including/excluding wrong columns (target column in test data, ID column)
   - Format mismatch with sample_submission.csv

Output ONLY the feedback as a concise numbered list.
"""

# ---------------------------------------------------------------------------
# Step 7 – Report (single-agent, no feedback loop)
# ---------------------------------------------------------------------------

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
