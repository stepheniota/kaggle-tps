# Kaggle TPS - July 2022

This month's **Tabular Playground Series** competition is Kaggle's first ever
unsupervised clustering challenge!
In the given dataset, each row belongs to a particular cluster.
Without any training data (no ground truth labels, no number of clusters), we
are tasked to predict which cluster each row belongs to.


## Evaulation
This competition's evaluation criterion is the
[Adjusted Rand Index](https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index)
between the (unknown) ground truth labels and our predicted cluster labels.

