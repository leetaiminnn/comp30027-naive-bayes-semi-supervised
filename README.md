# Naive Bayes from scratch + semi-supervised self-training

> 从零实现混合朴素贝叶斯并扩展为半监督自训练。COMP30027 Machine Learning（墨尔本大学，2026 S1）项目 1，个人完成。

Individual project for COMP30027 Machine Learning, University of Melbourne, Semester 1 2026.
Task: predict whether annual income exceeds $50K from the Adult (1994 US Census) data,
using a naive Bayes classifier written from first principles.

## What is in the notebook

1. **Supervised naive Bayes, implemented by hand.** Priors, Gaussian likelihoods for the
   continuous features and categorical likelihoods with Laplace smoothing for the discrete ones,
   combined in log space; out-of-vocabulary categories at test time are handled explicitly.
   Test accuracy **83.1%** on 15,087 held-out rows (macro-F1 0.75).
2. **Model evaluation.** Accuracy, confusion matrix, per-class precision / recall / F1 and a
   look at which features carry the signal.
3. **Semi-supervised extension.** Self-training: the supervised model labels 15,059 unlabelled
   rows, pseudo-labels are kept only when the class-probability ratio exceeds a threshold
   (R > 2, 5, 10, 20 compared), and the model is refitted on the enlarged training set.
4. **Comparison** of the supervised and semi-supervised models on the test set.

## Running it

Place the four course CSV files under `COMP30027_2026_Assignment1_data/Assignment1_data/`
and run the notebook top to bottom (Python 3.11+, numpy, pandas, scikit-learn for metrics
and encoding only, matplotlib). The data files are not included in this repository.

