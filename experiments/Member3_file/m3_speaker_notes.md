# Member 3 presentation script

## EDA at 1:00–2:00

We started with 1,698 tweets and retained 1,525 after cleaning. Each row represents a tweet. Positive is the largest class at 39.8%, while Neutral is the smallest at 28.4%. With this difference, we assess each class using Macro-F1 and class-level recall. The clearest feature pattern is text length. The median Neutral tweet has 86 characters, compared with 156 for Positive and 161 for Negative. The distributions overlap, so we evaluate length as a candidate feature. The hashtag and market plots also show how the collection filters shape the sample. These patterns apply to the tweets in this dataset.

## SHAP and business implications at 10:00–11:00

Adding the hand-crafted features to SBERT improved XGBoost’s outer-fold Macro-F1 from 0.527 to 0.579. The TF-IDF baseline scored 0.582. TreeSHAP points to character count, word count and hashtag ratio as the strongest named features, especially for Neutral. The model still recalled only 48.5% of Negative labels. In the team’s error review, the word Nifty appeared in a market-drop example labelled Positive. For a monitoring dashboard, I would show the matching text and confidence, route uncertain posts for review and check finance-specific edge cases with human reviewers.
