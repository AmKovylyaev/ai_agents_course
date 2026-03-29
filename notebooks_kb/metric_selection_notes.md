# Metric notes

- Binary classification: prioritize ROC-AUC unless competition metric differs.
- Regression: start with RMSE if leaderboard metric is unknown, then align with competition metric.
- Always compare local CV score and public leaderboard score.
