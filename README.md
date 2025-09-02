# Fair Supervised Learning Through Constraints on Smooth Nonconvex Unfairness-Measure Surrogates

This is the code implementation of our Fair Learning Framework.

## Running the code

The requirements are given in [requirements.txt].
* All dependencies can be installed via:
    * ``pip install -r requirements.txt``


* To generate all results from our experiments:
    * ``python main_train.py``
    * This will train models on the Dutch, Law School, and ACSIncome datasets and save results to CSV files in the `output/` directory.

* To generate all plots from the paper:
    * ``python main_plot.py``
    * This generates all figures in the `output/` directory based on the result files.

* To generate Variance analysis Results from our experiments:
    * ``python rep.py``
    * This will train models for different seeds (42, 123, 456, 789, 999) on the Dutch, Law School, and ACSIncome datasets and save results to CSV files in the `output/seeds` directory.

* To generate plots for the Variance analysis:
    * ``python plot_variance.py``
    * This generates all figures in the `output/plots` directory based on the result files in `output/seeds` directory.


## Key Files

* `fairness_models.py`: Fairness-constrained models
* `fairness_metrics.py`: Evaluation metrics
* `stochasticsqp.py`: StochasticSQP optimizer
