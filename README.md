# Customer Retention AI System

An end-to-end machine learning system that predicts customer churn and recommends retention actions.

## Problem
Customer churn leads to direct revenue loss. Companies need early signals to retain customers.

## Solution
This system predicts churn probability, segments customers into risk levels, and suggests retention actions.

## Features
- Churn probability prediction
- Risk segmentation (Low / Medium / High)
- Recommended business actions
- Live HTML UI

## Tech Stack
- Python
- FastAPI
- XGBoost
- Scikit-learn
- Railway

## Demo
Live UI: https://customer-churn-prediction-production-b1fe.up.railway.app/ui  

## How it works
Input → Preprocessing → Model → Probability → Risk → Action

## Notes
The HTML UI is served directly from `app.py` using FastAPI responses.
