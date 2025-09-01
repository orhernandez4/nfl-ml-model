# SWIFT (Sample-Weighted Inductive Football Transformer)

![SWIFT logo](https://i.imgur.com/IgZi9pt.png)

Welcome to the repo for SWIFT, a machine learning model designed to produce well-calibrated probabilities for NFL games.

Current status: trains a few models and tunes hyperparameters with the hyperopt library.

Development for this repo has been somewhat haphazard. I've now removed a significant number of features in order to take a more systematic approach to feature engineering, starting with game-level data.

I've also rewritten all of the data functions using polars, so it no longer takes 5 minutes to build all of the features. Yay.

## Getting started
There are two ways to get started:

1. Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/) for Python, then simply `bash run.sh` and you're off to the races!

2. Or, you can manage your own venv and install the dependencies in `requirements.txt`, then:
- Run `src/data/build.py` to build training and testing datasets
- Run `src/models/train.py` to train the model
- Run `src/data/predict/predict.py` to generate predictions for upcoming games

After training, you can view model scores in `/data/results`

If you want to generate predictions on upcoming games, specify the current NFL week and season in `/src/config/config.py`

## Documentation

For source code documentation, see `docs/build/html/index.html`

## FAQ
**Q: Is your model any good?**

A: Probably not.

**Q: Your acronym doesn't make any sense.**

A: That's not a question.

**Q: How come you don't use efficiency metrics like EPA/WPA?**

A: Data leakage! These metrics are derived from all currently available NFL data and would give the model an unfair glimpse of the future.

## Current engineered features
- home/away rest
- adjusted yards per play
- pythagorean expectation
- QB rating

## Training procedure
SWIFT does everything possible to avoid data leakage. It should never get a glimpse into the future.

#### Pipeline:
1. Transform the home/away team structure into an object/adversary team structure so that the model tries to predict a 50/50 mix of home and away win probabilities.
2. Train and evaluate using a grouped time series cross-validation scheme. The model trains on a block of *m* consecutive seasons, then validates on the following *n* seasons.
3. On each training fold in the time series cv, calibrate model probabilities with a 5-fold cv.
4. Search for optimal hyperparameters with hyperopt and minimize the average brier score.
5. Train final model on full dataset using optimal hyperparameters.
6. Evaluate on holdout data.

## Future tasks
- ~~Auto-generate API docs with Sphinx~~
- Complete documentation for all modules and functions
- Account for quarterback injuries
- Expand model evaluation to include tracking optimal hyperparameters
- Write bespoke time series cross validation windows that look forward **and** backward
- Use QLattice to engineer new QB rating feature
