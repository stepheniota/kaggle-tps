# Kaggle Tabular Playground Series - May 2022

This repo showcases my participation in the [May 2022 TPS](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/overview) Kaggle competition!
This month's TPS is a binary classification problem, an opportunity to explore various methods for identifying and exploiting feature interactions.

## Data
We are given (simulated) manufacturing control data and are asked to predict whether the machine is in state `0` or state `1`.

### Files
The datafiles can be downloaded using the `kaggle-cli`.
```
mkdir input && cd input
kaggle competitions download -c tabular-playground-series-may-2022
```
* `train.csv` - the training data, which includes normalized continuous data and categorical data
* `test.csv` - the test set; your task is to predict binary target variable which represents the state of a manufacturing process
* `sample_submission.csv` - a sample submission file in the correct format


## Evaluation
Submissions are evaluated on [area under the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

<!--
### Submission File
Each id in the test set much have a corresponding predicted probability for the target variable. The submission file should contain a header and have the following format.
```
id,target
900000,0.65
900001,0.97
900002,0.02
...
```
-->
