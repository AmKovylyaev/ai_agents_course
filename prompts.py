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
- Feedback from judge (if any): {improvement_hint}

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
Do NOT include any code, code snippets, or code blocks.
Be concise — state what to do, not why.
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
  numeric_columns (must exclude the target column),
  categorical_columns (must exclude the target column),
  target_column (string), missing_values, n_classes (integer),
  task_type ("regression" or "classification")

Task type detection:
  - If the target is numeric and has many unique values (>20, or unique ratio > 5%% of rows), treat it as regression.
  - Otherwise, treat it as classification.

Save the EDA report to session_dir/reports/eda_summary.txt.
Save ONLY these graphs to session_dir/reports/:
  - target_distribution.png — bar chart (classification) or histogram (regression)
  - correlation_heatmap.png — heatmap of numeric feature correlations
  - missing_values.png — bar chart of missing values per column (skip if no missing values)
Do not create per-column histograms, KDE plots, or other extra graphs.

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

Context:
- Train data is at TRAIN_DATA_PATH
- Outputs go to SESSION_DIR
- Train/val split fraction: {train_sample_frac} (random_state=42)
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}
- Feedback from judge (if any): {improvement_hint}

IMPORTANT — Use ALL available columns with appropriate encoding:
- FIRST: drop the target column to get X; derive column lists from X (not from the full DataFrame).
- Build a sklearn Pipeline with ColumnTransformer. Split categoricals by cardinality:
  * Numeric columns: SimpleImputer(strategy="median") → StandardScaler
  * Low-cardinality categoricals (≤50 unique): SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore", sparse_output=False)
  * High-cardinality categoricals (>50 unique): SimpleImputer(strategy="constant", fill_value="__missing__") → OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

Choose a model appropriate for the task_type. Consider the data size, number of features, and the nature of the target. Training must complete within 2 minutes.
For the first iteration (no judge feedback), default to CatBoost. On subsequent iterations, try a different model based on judge suggestions.

Available libraries: sklearn, catboost, xgboost, lightgbm, category_encoders, pandas, numpy.

Save the ENTIRE Pipeline object (preprocessing + model) to SESSION_DIR/models/pipeline.joblib

Output a numbered step-by-step plan covering:
1. Load data
2. Identify and separate the target column
3. Target transformation (if regression)
4. Classify feature columns by type and cardinality
5. Design the preprocessing (ColumnTransformer)
6. Choose a model
7. Assemble the full Pipeline (preprocessor + model)
8. Split, fit, and report initial metrics
9. Save the pipeline
10. State dict keys to update (including target_transform)

If there was a previous error or verifier feedback, adjust the plan accordingly.
Output ONLY the plan as a numbered list in plain English.
Do NOT include any code, code snippets, or code blocks.
Be concise — state what to do, not why.
"""

STEP2_TRAIN_PROMPT = """Follow the plan below. Write complete Python code, then execute it using the run_code tool.

Plan:
{plan}

Paths:
- Train data: {train_path}
- Session directory: {session_dir}
- Train/val split fraction: {train_sample_frac} (random_state=42)

EDA results:
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}
- Target column: {target_column}
- Number of unique target values: {n_classes}
- Task type: {task_type}

Previous attempt (if retry):
- Error: {last_error}
- Previous code: {previous_code}
- Verifier feedback: {verifier_feedback}

Requirements:
- Drop the target column first, then verify which EDA-reported feature columns actually exist in the DataFrame. Only use columns present in the data.
- For regression: apply log1p to the target before fitting and set state["target_transform"] = "log1p".
- Build a sklearn Pipeline (ColumnTransformer + model) and save the entire pipeline to session_dir/models/pipeline.joblib.
- The pipeline must handle raw DataFrame columns directly (all preprocessing inside the pipeline).
- Do not define custom transformer classes or create separate Python modules. Use only well-known library components (sklearn, catboost, xgboost, lightgbm, category_encoders, etc.) so the saved pipeline loads cleanly in any process.
- Print metrics after fitting.
- Update state dict with: target_column, model_path, X_train_shape, X_val_shape, model_type, task_type, target_transform.

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
2. Call check_model_file — does the pipeline file exist and is it loadable?
3. Use read_session_file to inspect the saved code if needed

Watch for:
- Target column included in feature columns
- Pipeline not saved as a complete object (only the model saved separately)
- Custom transformer classes that break serialisation
- Model type mismatched with task_type (e.g. classifier for regression)

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
- Improvement hints from previous judge (if any): {improvement_hint}

Key points:
- The saved pipeline already includes all preprocessing — pass raw feature columns directly.
- Reproduce the exact same train/val split used during training (fraction {train_sample_frac}, random_state=42).
- If the target was log-transformed during training (state["target_transform"] == "log1p"), apply the same transformation to the true targets before splitting, then inverse-transform both predictions and targets before computing metrics. All metrics should be on the original scale.

Output a numbered step-by-step plan covering:
1. Load the pipeline and the training data
2. Separate features and target
3. Handle target transformation if applicable (to reproduce the same split)
4. Recreate the train/val split
5. Predict on both train and val sets
6. Inverse-transform predictions and targets if needed
7. Compute appropriate metrics for both train and val sets based on task_type
   - Regression: MSE, MAE, R²
   - Classification: accuracy, precision (macro), recall (macro), f1 (macro)
8. Store as a nested dict with "train" and "val" sub-dicts
9. Save metrics to SESSION_DIR/reports/local_metrics.json
10. Print all metrics clearly
11. Update state dict with local_metrics

If there was a previous error or verifier feedback, adjust the plan accordingly.
Output ONLY the plan as a numbered list in plain English.
Do NOT include any code, code snippets, or code blocks.
Be concise — state what to do, not why.
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
- Train/val split fraction: {train_sample_frac} (random_state=42)

Previous attempt (if retry):
- Error: {last_error}
- Previous code: {previous_code}
- Verifier feedback: {verifier_feedback}

Requirements:
- The saved pipeline handles all preprocessing — pass raw feature columns directly.
- Reproduce the exact train/val split (fraction {train_sample_frac}, random_state=42).
- If state["target_transform"] == "log1p", apply the same log1p to targets before splitting (to get identical indices), then inverse-transform both predictions and true targets before computing metrics. All metrics must be on the original scale.
- Predict on BOTH train and val sets.
- Compute appropriate metrics for the task_type:
  * Regression: MSE, MAE, R² (keys: "mse", "mae", "r2")
  * Classification: accuracy, precision, recall, f1 (macro average; keys: "accuracy", "precision", "recall", "f1")
- Store as nested dict: {{"train": {{...}}, "val": {{...}}}}
- Save to session_dir/reports/local_metrics.json.
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
4. Verify local_metrics has both "train" and "val" sub-dicts with the correct metric keys

Watch for:
- Manual preprocessing instead of using the pipeline directly
- Wrong split reproduction (different random_state or fraction)
- Metrics computed on transformed scale instead of original scale
- Wrong metric type for the task (classification metrics for regression or vice versa)
- Missing train or val metrics (both are required)

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
- Sample submission is at SAMPLE_SUBMISSION_PATH (defines the exact output format)
- Target column name: {target_column}
- Task type: {task_type}
- Outputs go to SESSION_DIR
- Previous error (if retry): {last_error}
- Verifier feedback (if retry): {verifier_feedback}

Key points:
- The pipeline handles all preprocessing — pass raw feature columns directly.
- If the target was log-transformed during training (state["target_transform"] == "log1p"), inverse-transform the predictions to restore the original scale.
- The submission must match the sample submission exactly: same columns, same order, same number of rows.

Output a numbered step-by-step plan covering:
1. Load the sample submission to learn the expected format
2. Load the pipeline and the test data
3. Remove the target column from test data if present
4. Generate predictions using the pipeline
5. Inverse-transform predictions if target was log-transformed
6. Assemble the output DataFrame matching the sample submission format
7. Verify column names and row count match the sample
8. Save to SESSION_DIR/submission.csv
9. Update state dict

If there was a previous error or verifier feedback, adjust the plan accordingly.
Output ONLY the plan as a numbered list in plain English.
Do NOT include any code, code snippets, or code blocks.
Be concise — state what to do, not why.
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

Requirements:
- The pipeline handles all preprocessing — pass raw feature columns directly.
- Remove the target column from test data before predicting (it may not be present, so handle gracefully).
- If state["target_transform"] == "log1p", the pipeline outputs log-scale values — inverse-transform predictions to restore the original scale.
- Load the sample submission first to learn the expected column names and row count.
- The output CSV must match the sample submission exactly: same columns, same order, same number of rows. Verify this before saving.
- Save to session_dir/submission.csv without the index.
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

Watch for:
- Column or row count mismatch with the sample submission
- NaN or missing predictions
- Predictions still in log-scale when target was log-transformed

Based on tool results, provide your verdict:
- If ALL checks pass: summarize what passed, state "APPROVED"
- If ANY check fails: list each failure with a specific fix, state "FAIL"
"""

# ---------------------------------------------------------------------------
# Step – Judge (LLM-as-a-judge, outer refinement loop)
# ---------------------------------------------------------------------------

STEP_JUDGE_PROMPT = """You are a Judge evaluating ML experiment results. Be extremely concise.

Metrics (train / val): {local_metrics}
Model info: {previous_code}

Rules:
- SUFFICIENT if val metrics are reasonable and no severe overfitting (train/val gap < 2x).
- NEED_REFINEMENT otherwise.
- Each suggestion must be ONE short sentence — a single concrete action, not a list.
- Do NOT suggest hyperparameter tuning, grid search, or random search.

Output ONLY a JSON object:
{{"decision": "SUFFICIENT or NEED_REFINEMENT", "reasoning": "one sentence", "eda_suggestions": "one sentence or empty", "train_suggestions": "one sentence or empty"}}
"""

# ---------------------------------------------------------------------------
# Step 7 – Report (single-agent, no feedback loop)
# ---------------------------------------------------------------------------

STEP7_REPORT_PROMPT = """You are an ML engineer. Write Python code to generate a final report.

Context:
- Session directory: {session_dir}
- EDA insights: session_dir/reports/eda_summary.txt
- Local metrics: session_dir/reports/local_metrics.json
- Task type: {task_type}
- Submission status: {submit_ok}
- Kaggle scores: public={public_score}, private={private_score}

Requirements:
1. Load EDA summary and local metrics from their files (if they exist)
2. Compile all results into a structured report covering:
   - Task type
   - EDA insights
   - Model parameters
   - Local validation metrics (both train and val if available)
   - Kaggle submission results
   - Lessons learned and next steps
3. Save as JSON (session_dir/reports/final_report.json) and as readable text (session_dir/reports/final_report.txt)
4. Update state dict with: report_path

Output ONLY executable Python code.
"""
