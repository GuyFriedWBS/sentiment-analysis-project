# Sentiment Analysis Project

## Project Description
This project trains a simple sentiment analysis model using logistic regression and TF-IDF vectorization.  
The model predicts whether a text is positive (1) or negative (0).

## Setup

### Option 1: Python venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### Option 2: Conda
conda create -n sentiment-env python=3.11 -y
conda activate sentiment-env
pip install -r requirements.txt

## Train
python src/train.py --data data/train.csv --out models/sentiment.joblib
