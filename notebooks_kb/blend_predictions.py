import pandas as pd

sub1 = pd.read_csv("submission_model1.csv")
sub2 = pd.read_csv("submission_model2.csv")

target_col = sub1.columns[-1]
blend = sub1.copy()
blend[target_col] = 0.5 * sub1[target_col] + 0.5 * sub2[target_col]
blend.to_csv("submission_blend.csv", index=False)
