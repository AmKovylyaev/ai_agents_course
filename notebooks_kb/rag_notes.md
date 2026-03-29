# Notes for notebook RAG

Useful retrieval keywords:
- tabular regression baseline
- binary classification pipeline
- categorical preprocessing
- ColumnTransformer
- Pipeline
- cross validation
- submission creation

Recommended implementation patterns:
1. Detect numeric and categorical columns separately.
2. Use SimpleImputer for both groups.
3. Use OneHotEncoder(handle_unknown="ignore") for safe inference.
4. Keep preprocessing and model inside one Pipeline.
5. Save submission in Kaggle format with the same column order as sample submission.