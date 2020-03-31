# Calibrated Precision

This repository provides python codes to reproduce the experimental results from paper **["Master your Metrics with Calibration" Wissam Siblini, Jordan Fréry, Liyun He-Guelton, Frédéric Oblé and Yi-Qing Wang (2019)](https://arxiv.org/abs/1909.02827).** 

It includes the implementation of the calibrated precision as well as the calibrated f-score (in calibrated_metrics.py), the calibrated average precision (calibrated_metrics.py) and the calibrated precision-recall gain (prgc.py). It also includes a notebook that allows to reproduce all the results from the paper.

## Usage

### Calibrated f-score, calibrated average precision

The calibated average-precision (or aucpr) is based on **[scikit-learn's implementation of the average-precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)**) and is implemented as an extra parameter:

```
average_precision(y_true, y_pred, pos_label=1, sample_weight=None,pi0=None)
```

If pi0 is None, the function computes the regular average precision. Otherwise it computes the calibrated average precision with parameter pi0. 

The calibated f1-score is also implemented as a special case : 

```
f1score(y_true, y_pred, pi0=None)
```
### paper_experiments.ipynb

paper_experiments.ipynb is a notebook with 4 sections : 

* Experiments with synthetic data : it runs the experiments with synthetic data from the paper (one experiments that shows invariance of calibrated metrics wrt the positive class ration and one experiment that show that they still assess model performance)

(insert figure here)

* Experiment with real data : it runs an experiment equivalent to the invariance experiment on **[a real world imbalanced dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)**
* Comparison between proposed formula and heuristic :
* Experiments on openml :


## Citation

If you use the code in this repository, please cite :

```
@article{siblini2019master,
  title={Master your Metrics with Calibration},
  author={Siblini, Wissam and Fr{\'e}ry, Jordan and He-Guelton, Liyun and Obl{\'e}, Fr{\'e}d{\'e}ric and Wang, Yi-Qing},
  journal={arXiv preprint arXiv:1909.02827},
  year={2019}
}
```
