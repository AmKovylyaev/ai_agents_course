"""LLM prompt templates for each pipeline step.

Architecture:
  - Planner: simple LLM chain (no tools) — plain-English plan only
  - Coder: ReAct agent with tools (validate_syntax, check_safety, run_code)
  - Verifier: ReAct agent with guardrail tools — runs after EVERY attempt

The pipeline auto-detects task_type (regression vs classification) during EDA.
All subsequent steps adapt models, metrics, and evaluation accordingly.

train_path always points to the ready-to-use training CSV — steps just load it directly.
"""

# ---------------------------------------------------------------------------
# Step 1 – EDA
# ---------------------------------------------------------------------------

STEP1_PLANNER_PROMPT = """You are an ML engineering planner. Create a concise plan for Exploratory Data Analysis.

Context:
- Train data path: {train_path}
- Test data path: {test_path}
- Session directory: {session_dir}
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}

Output a numbered step-by-step plan covering:
1. Load the train and test data from the provided paths
2. Which statistics to compute for ALL columns (shape, dtypes, missing values, unique counts)
3. How to profile numeric columns (min, max, mean, std, correlations)
4. How to profile categorical/text columns (cardinality, top values)
5. Determine the task type:
   - If the target column is numeric AND has many unique values relative to the dataset size
     (e.g. >20 unique values, or unique ratio > 5%), it is REGRESSION
   - Otherwise, it is CLASSIFICATION
   - Save task_type as "regression" or "classification" in state
6. Target variable analysis (distribution, class balance for classification; histogram + stats for regression)
7. Save ONLY these graphs to SESSION_DIR/reports/ (use matplotlib, close each figure after saving):
   - target_distribution.png — bar chart (classification) or histogram (regression) of target
   - correlation_heatmap.png — heatmap of numeric feature correlations
   - missing_values.png — bar chart of missing value counts per column (only if there are missing values)
   Do NOT create per-column plots or any other graphs.
8. Save text report to SESSION_DIR/reports/eda_summary.txt
9. Which keys to add to the state dict:
   - train_shape, test_shape, columns
   - numeric_columns (list of numeric feature column names, EXCLUDING the target column)
   - categorical_columns (list of categorical feature column names, EXCLUDING the target column)
   - target_column, missing_values, n_classes (number of unique target values)
   - task_type ("regression" or "classification")

If there was a previous error or verifier feedback, adjust the plan to avoid the same mistake.
Output ONLY the plan as a numbered list in plain English.
Do NOT include any code, code snippets, or code blocks — describe what to do, not how to code it.
"""

STEP1_EDA_PROMPT = """Follow the plan below. Write complete Python code, then execute it using the run_code tool.

Plan:
{plan}

Paths:
- Train data: {train_path}
- Test data: {test_path}
- Session directory: {session_dir}

Previous attempt (if retry):
- Error: {last_error}
- Previous code: {previous_code}
- Verifier feedback: {verifier_feedback}

The state dict MUST be updated with ALL of these keys:
  train_shape, test_shape, columns,
  numeric_columns (MUST NOT include the target column),
  categorical_columns (MUST NOT include the target column),
  target_column (string), missing_values, n_classes (integer),
  task_type ("regression" or "classification")
  
Your code MUST update the state dictionary with all of the following keys:
- target_column
- numeric_columns
- categorical_columns
- task_type
- n_classes

If any of these keys are missing or empty, the step is considered failed.

The code must:
1. Load the train dataframe from state.
2. Detect the target column.
3. Split features into numeric and categorical columns.
4. Infer task_type as either "classification" or "regression".
5. Set n_classes (for regression use 1).
6. Save all results back into state.

Task type detection rule:
  - If target is numeric AND has >20 unique values (or unique ratio > 5%% of rows): task_type = "regression"
  - Otherwise: task_type = "classification"

Save the EDA report to session_dir/reports/eda_summary.txt.
Save ONLY these graphs to session_dir/reports/ (use matplotlib, call plt.close() after each):
  - target_distribution.png — bar chart (classification) or histogram (regression)
  - correlation_heatmap.png — heatmap of numeric feature correlations
  - missing_values.png — bar chart of missing values per column (skip if no missing values)
Do NOT create per-column histograms, KDE plots, or any other extra graphs.

IMPORTANT: You MUST call the run_code tool with your complete code. Do NOT just output the code as text.
"""

STEP1_VERIFIER_PROMPT = """Verify the EDA step results. Use your guardrail tools to check ALL requirements.

Code that was executed:
{previous_code}

Execution result / error:
{last_error}

Plan that was followed:
{plan}

Context:
- Train data path: {train_path}
- Test data path: {test_path}
- Session directory: {session_dir}

Verification checklist:
1. Call check_state_completeness — are all required keys populated (target_column, numeric_columns, categorical_columns, n_classes, train_shape)?
2. Call check_data_leakage — is the target column absent from numeric_columns and categorical_columns?
3. Use read_session_file to check that the EDA report was saved

Based on tool results, provide your verdict:
- If ALL checks pass: summarize what passed, state "APPROVED"
- If ANY check fails: list each failure with a specific fix, state "FAIL"
"""

# ---------------------------------------------------------------------------
# Step 2 – Train
# ---------------------------------------------------------------------------

STEP2_PLANNER_PROMPT = """You are an ML engineering planner. Create a concise plan for training a model.

EDA results from step 1 (available in state):
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}
- Target column: {target_column}
- Number of unique target values: {n_classes}
- Train shape: {train_shape}
- Task type: {task_type}

Notebook RAG context:
- Query used: {rag_query}
- Retrieved notebook context:
{rag_context}

Context:
- Train data is at TRAIN_DATA_PATH
- Outputs go to SESSION_DIR
- Train/val split: test_size={train_sample_frac}, random_state=42
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}

IMPORTANT — Use ALL available columns with appropriate encoding:
- FIRST: drop the target column to get X; derive column lists from X (not from the full DataFrame).
- Build a sklearn Pipeline with ColumnTransformer. Split categoricals by cardinality:
  * Numeric columns: SimpleImputer(strategy="median") → StandardScaler
  * Low-cardinality categoricals (≤50 unique): SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore", sparse_output=False)
  * High-cardinality categoricals (>50 unique): SimpleImputer(strategy="constant", fill_value="__missing__") → OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

Choose a model based on task_type:
- If REGRESSION:
  * RandomForestRegressor(n_estimators=100, random_state=42) — fast, solid default
  * Use regression metrics: RMSE, MAE, R²
- If CLASSIFICATION:
  * RandomForestClassifier(n_estimators=100, random_state=42) — fast, solid default
  * Use classification metrics: accuracy, precision, recall, f1

Save the ENTIRE Pipeline object (preprocessing + model) to SESSION_DIR/models/pipeline.joblib

Output a numbered step-by-step plan covering:
1. Load data from TRAIN_DATA_PATH
2. Target column identification; drop target from X
3. Splitting numeric vs categorical feature columns (from X, NOT from the full DataFrame)
4. Splitting categoricals into low-cardinality (≤50 unique) and high-cardinality (>50)
5. Building the ColumnTransformer: numeric, low-card OneHotEncoder, high-card OrdinalEncoder
6. Model selection based on task_type and hyperparameters (must train in <2 min)
7. Assembling the full Pipeline (preprocessor + model)
8. Train/val split (test_size={train_sample_frac}), fitting, and metric reporting
9. Saving the pipeline to SESSION_DIR/models/pipeline.joblib
10. State dict keys to update

Use the retrieved notebook context as inspiration for:
- preprocessing choices
- pipeline structure
- feature handling
- training/evaluation patterns

Do NOT blindly copy broken code.
Reuse only ideas and reliable implementation patterns.

If there was a previous error or verifier feedback, adjust the plan accordingly.
Output ONLY the plan as a numbered list in plain English.
Do NOT include any code, code snippets, or code blocks — describe what to do, not how to code it.
"""

STEP2_TRAIN_PROMPT = """Follow the plan below. Write complete Python code, then execute it using the run_code tool.

Plan:
{plan}

Paths:
- Train data: {train_path}
- Session directory: {session_dir}
- Train/val split: test_size={train_sample_frac}, random_state=42

EDA results:
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}
- Target column: {target_column}
- Number of unique target values: {n_classes}
- Task type: {task_type}

Notebook RAG context:
- Query used: {rag_query}
- Search type: {rag_search_type}
- Retrieved notebook context:
{rag_context}

Use the retrieved notebook examples as implementation guidance.
Prefer reusable patterns that match the current dataset schema and task type.
Do not blindly copy code that conflicts with available columns or target structure.

Previous attempt (if retry):
- Error: {last_error}
- Previous code: {previous_code}
- Verifier feedback: {verifier_feedback}

Hard constraints:
- FIRST drop the target column from the DataFrame to get X, then derive numeric_cols and categorical_cols from X (do NOT trust the EDA lists blindly — they may include the target). Filter: numeric_cols = [c for c in numeric_cols if c in X.columns]; same for categorical_cols.
- Build a sklearn Pipeline (ColumnTransformer + model) — the ENTIRE pipeline must be saved as a single joblib file to session_dir/models/pipeline.joblib.
- Fit on raw DataFrame columns (do NOT pre-transform before passing to pipeline).
- Choose model based on task_type:
  * regression → RandomForestRegressor; print RMSE, MAE, R²
  * classification → RandomForestClassifier; print accuracy, precision, recall, f1
- Update state dict with: target_column, model_path, X_train_shape, X_val_shape, model_type, task_type
- Use the retrieved notebook context as implementation guidance, especially for:
  * preprocessing structure
  * ColumnTransformer + Pipeline usage
  * handling categorical features
  * training/evaluation flow
- Do NOT rely on notebook code that contradicts the current dataset schema or task_type.
- Prefer CatBoostRegressor for regression tasks when categorical features are present.
- Prefer CatBoostClassifier for classification tasks when categorical features are present.
- Avoid RandomForest as the primary model unless there is a strong justification.
- CatBoost can consume categorical columns directly via cat_features.
- Save the trained model to state["model_path"] using joblib.
- Ensure the model is compatible with later inference on the test set.

IMPORTANT: You MUST call the run_code tool with your complete code. Do NOT just output the code as text.
"""

STEP2_VERIFIER_PROMPT = """Verify the training step results. Use your guardrail tools to check ALL requirements.

Code that was executed:
{previous_code}

Execution result / error:
{last_error}

Plan that was followed:
{plan}

Context:
- Train data path: {train_path}
- Session directory: {session_dir}
- Task type: {task_type}

Verification checklist:
1. Call check_state_completeness — is model_path populated?
2. Call check_model_file — does the pipeline file exist and is it loadable with a predict() method?
3. Use read_session_file to inspect the saved code if needed

Common pitfalls to check for in the code:
- Not using ColumnTransformer / Pipeline correctly
- Including the target column in feature columns
- sparse_output not set to False in OneHotEncoder
- Not saving the full pipeline (only saving the model)
- Using a Classifier for regression or a Regressor for classification

Based on tool results, provide your verdict:
- If ALL checks pass: summarize what passed, state "APPROVED"
- If ANY check fails: list each failure with a specific fix, state "FAIL"
"""

# ---------------------------------------------------------------------------
# Step 3 – Eval
# ---------------------------------------------------------------------------

STEP3_PLANNER_PROMPT = """You are an ML engineering planner. Create a concise plan for local model evaluation.

Context:
- Train data is at TRAIN_DATA_PATH
- Saved pipeline (preprocessing + model) is at MODEL_PATH
- Outputs go to SESSION_DIR
- Task type: {task_type}
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}

Key point:
- The saved pipeline.joblib contains the FULL sklearn Pipeline (ColumnTransformer + model).
- You do NOT need to preprocess the data manually — just pass raw DataFrame columns to pipeline.predict().
- Reproduce the exact train/val split: train_test_split(test_size=TRAIN_SAMPLE_FRAC, random_state=42).

Output a numbered step-by-step plan covering:
1. Load the pipeline from MODEL_PATH
2. Load the training data from TRAIN_DATA_PATH
3. Identify target column, separate X and y
4. Recreate the train/val split (test_size=TRAIN_SAMPLE_FRAC, random_state=42) to get the validation set
5. Call pipeline.predict(X_val) — NO manual preprocessing
6. Compute metrics based on task_type:
   - If REGRESSION: RMSE, MAE, R²
   - If CLASSIFICATION: accuracy, precision (macro), recall (macro), f1 (macro)
7. Save metrics dict to SESSION_DIR/reports/local_metrics.json
8. Print all metrics clearly
9. Update state dict with local_metrics

If there was a previous error or verifier feedback, adjust the plan accordingly.
Output ONLY the plan as a numbered list in plain English.
Do NOT include any code, code snippets, or code blocks — describe what to do, not how to code it.
"""

STEP3_EVAL_PROMPT = """Follow the plan below. Write complete Python code, then execute it using the run_code tool.

Plan:
{plan}

Paths:
- Train data: {train_path}
- Pipeline (preprocessing + model): {model_path}
- Session directory: {session_dir}
- Target column: {target_column}
- Task type: {task_type}
- Train/val split: test_size={train_sample_frac}, random_state=42

Previous attempt (if retry):
- Error: {last_error}
- Previous code: {previous_code}
- Verifier feedback: {verifier_feedback}

Hard constraints:
- The saved pipeline.joblib is a FULL sklearn Pipeline — pass RAW DataFrame columns to pipeline.predict(), do NOT manually preprocess.
- Reproduce the exact train/val split: train_test_split(test_size={train_sample_frac}, random_state=42).
- Save metrics to session_dir/reports/local_metrics.json.
- Compute metrics based on task_type:
  * If regression: compute rmse (root mean squared error), mae (mean absolute error), r2 (R² score).
    Store as dict with keys: "rmse", "mae", "r2".
  * If classification: compute accuracy, precision_score(average="macro"), recall_score(average="macro"), f1_score(average="macro").
    Store as dict with keys: "accuracy", "precision", "recall", "f1".
- Print all metrics clearly.
- Update state dict with: local_metrics

IMPORTANT: You MUST call the run_code tool with your complete code. Do NOT just output the code as text.
"""

STEP3_VERIFIER_PROMPT = """Verify the evaluation step results. Use your guardrail tools to check ALL requirements.

Code that was executed:
{previous_code}

Execution result / error:
{last_error}

Plan that was followed:
{plan}

Context:
- Train data path: {train_path}
- Model path (pipeline): {model_path}
- Session directory: {session_dir}
- Task type: {task_type}

Verification checklist:
1. Call check_state_completeness — is local_metrics populated?
2. Call check_metrics — are all expected metrics present and valid?
3. Use read_session_file to check that local_metrics.json was saved to session_dir/reports/

Common pitfalls:
- Manually preprocessing data instead of passing raw columns to pipeline.predict()
- Wrong data split reproduction (different random_state or fraction)
- Dropping columns that the pipeline expects
- Using classification metrics for a regression task or vice versa

Based on tool results, provide your verdict:
- If ALL checks pass: summarize what passed, state "APPROVED"
- If ANY check fails: list each failure with a specific fix, state "FAIL"
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
- Task type: {task_type}
- Outputs go to SESSION_DIR
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}

Key points:
- The saved pipeline.joblib contains the FULL sklearn Pipeline (ColumnTransformer + model).
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
Output ONLY the plan as a numbered list in plain English.
Do NOT include any code, code snippets, or code blocks — describe what to do, not how to code it.
"""

STEP4_SUBMISSION_PROMPT = """Follow the plan below. Write complete Python code, then execute it using the run_code tool.

Plan:
{plan}

Paths:
- Pipeline (preprocessing + model): {model_path}
- Test data: {test_path}
- Sample submission: {sample_submission_path}
- Session directory: {session_dir}
- Target column: {target_column}
- Task type: {task_type}

Previous attempt (if retry):
- Error: {last_error}
- Previous code: {previous_code}
- Verifier feedback: {verifier_feedback}

Hard constraints:
- The saved pipeline.joblib is a FULL sklearn Pipeline — pass RAW DataFrame columns to pipeline.predict(), do NOT manually preprocess.
- Drop the target column ("{target_column}") from test data before predicting — use errors="ignore" since test data may not have this column.
- Load sample_submission.csv FIRST to get its exact column names.
- The output CSV MUST contain EXACTLY the same columns as sample_submission.csv — no extra columns, no missing columns. Use: out_df = sample[sample.columns].copy(), then replace the last column with predictions.
- Assert that output has the same number of columns and rows as the sample before saving.
- Save to session_dir/submission.csv with index=False.
- Update state dict with: submission_path

IMPORTANT: You MUST call the run_code tool with your complete code. Do NOT just output the code as text.
"""

STEP4_VERIFIER_PROMPT = """Verify the submission step results. Use your guardrail tools to check ALL requirements.

Code that was executed:
{previous_code}

Execution result / error:
{last_error}

Plan that was followed:
{plan}

Context:
- Model path (pipeline): {model_path}
- Test data path: {test_path}
- Sample submission path: {sample_submission_path}
- Session directory: {session_dir}
- Task type: {task_type}

Verification checklist:
1. Call check_state_completeness — is submission_path populated?
2. Call check_submission — does the submission CSV match the sample format (same columns, same row count, no NaN predictions)?
3. Use read_session_file to spot-check the submission file if needed

Common pitfalls:
- Manually preprocessing instead of using pipeline.predict()
- Including/excluding wrong columns (target column in test data, ID column)
- Format mismatch with sample_submission.csv

Based on tool results, provide your verdict:
- If ALL checks pass: summarize what passed, state "APPROVED"
- If ANY check fails: list each failure with a specific fix, state "FAIL"
"""

# ---------------------------------------------------------------------------
# Step 7 – Report (single-agent, no feedback loop)
# ---------------------------------------------------------------------------

STEP7_REPORT_PROMPT = """You are an ML engineer. Write Python code to generate a final report.

Context:
- Session directory: {session_dir}
- EDA insights available in: session_dir/reports/eda_summary.txt
- Local metrics available in: session_dir/reports/local_metrics.json
- Task type: {task_type}
- Submission status: {submit_ok}
- Kaggle scores: public={public_score}, private={private_score}

Requirements:
1. Load EDA summary from file if exists
2. Load local metrics from file if exists
3. Compile all results into a structured report
4. Save as JSON: session_dir/reports/final_report.json
5. Save as readable text: session_dir/reports/final_report.txt
6. Update state dict with: report_path (path to final_report.txt)
7. Include in the report:
   - Task type (regression or classification)
   - EDA insights
   - Model parameters
   - Local validation metrics
   - Kaggle submission results
   - Lessons learned and next steps

Output ONLY executable Python code in a ```python code block.
Use json, pathlib.
"""
