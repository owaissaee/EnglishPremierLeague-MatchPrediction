# EPL Match Result Prediction - CS446 Machine Learning

This repository contains code, figures, and a LaTeX report for predicting English Premier League match outcomes using machine learning.

## Task
Predict the match result as one of three classes:

- Home Win (H)
- Draw (D)
- Away Win (A)

## Models

- KNN
- Logistic Regression
- Naive Bayes
- SVM (RBF)
- Neural Network

## Main Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| KNN | 0.4974 | 0.4191 |
| Logistic Regression | 0.5123 | 0.3744 |
| Naive Bayes | 0.4956 | 0.3927 |
| SVM | 0.5158 | 0.3767 |
| Neural Network | 0.4289 | 0.3965 |
| Bet365 Odds | 0.5456 | 0.4044 |

Best trained classifier: **SVM (RBF)**.

## Folder Structure

```text
src/
  epl.py              # training, testing, CSV results
  prediction.py       # separate PNG plots
figures/              # generated PNG figures
report/
  main.tex            # LaTeX report
notebook/
  EPL_Match_Prediction_Notebook.ipynb
```

## How to Run

```bash
pip install -r requirements.txt
python src/epl.py
python src/prediction.py
```

Figures will be saved in:

```text
/home/owais-saeed/python/ML/EPL/figures/
```

## Report

Compile the LaTeX report:

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```
