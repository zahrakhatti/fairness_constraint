# Fair Supervised Learning Through Constraints on Smooth Nonconvex Unfairness-Measure Surrogates

A novel framework for training fair machine learning models using hard constraints on accurate, smooth nonconvex surrogates of unfairness measures, ensuring prescribed fairness bounds are actually achieved in practice.

## Problem Statement

Traditional fair ML approaches face critical limitations: loose convex surrogates fail to guarantee actual fairness, regularization methods require expensive hyperparameter tuning, and most approaches struggle with multiple conflicting fairness criteria. This framework introduces tight, smooth nonconvex surrogates with hard constraint enforcement to solve these challenges.


## Running the code

The requirements are given in [requirements.txt].

* All dependencies can be installed via:
    * `pip install -r requirements.txt`
* To generate all results from our experiments:
    * `python main_train.py`
    * This will train models on the Dutch, Law School, and ACSIncome datasets and save results to CSV files in the `output/` directory.
* To generate all plots from the paper:
    * `python main_plot.py`
    * This generates all figures in the `output/` directory based on the result files.
* To generate Variance analysis Results from our experiments:
    * `python rep.py`
    * This will train models for different seeds (42, 123, 456, 789, 999) on the Dutch, Law School, and ACSIncome datasets and save results to CSV files in the `output/seeds` directory.
* To generate plots for the Variance analysis:
    * `python plot_variance.py`
    * This generates all figures in the `output/plots` directory based on the result files in `output/seeds` directory.

## Key Files

* `fairness_models.py`: Fairness-constrained models
* `fairness_metrics.py`: Evaluation metrics
* `stochasticsqp.py`: StochasticSQP optimizer

## Key Results

* **Precise fairness control**: Desired δ = 0.8 → actual disparate impact ≈ 0.8
* **High accuracy maintenance**: <5% accuracy loss even with tight fairness constraints
* **Multi-fairness capability**: Simultaneous enforcement of demographic parity + disparate impact
* **No hyperparameter tuning**: Hard constraints eliminate expensive regularization parameter search
