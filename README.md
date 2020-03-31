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

The calibated f1-score is also implemented as a special case of the f1-score : 

```
f1score(y_true, y_pred, pi0=None)
```

### Calibrated pr-gain

Calibrated precision gain, recall gain and auc-pr-gain are based on the implementation of the **[regular ones](https://github.com/meeliskull/prg)**. Calib auc-pr-gain can be computed as follows :

```
import prgc
prgc.calc_auprg(prgc.create_prg_curve(y_true,y_pred,pi0))
```

### paper_experiments.ipynb

paper_experiments.ipynb is the notebook that contains the code the experiments from the paper. It has 4 sections : 

* Experiments with synthetic data : it runs the experiments with synthetic data (one experiments that shows invariance of calibrated metrics wrt the positive class ration and one experiment that show that they still assess model performance)

(insert figure here)

* Experiment with real data : it runs an experiment equivalent to the invariance experiment on **[a real world imbalanced dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)**

(insert figure here)

* Comparison between proposed formula and heuristic : This experiments shows empirically that the calibrated precision with parameter pi0 is equivalent to the precision that would be obtained if the positive class ratio pi was equal to pi0.

(insert figure here)

* Experiments on openml : The experiments in this section shows how calibration and the choice of metric impacts the selection of the best model. It empirically analyzes the correlation of several metrics in terms of model ordering. We use OpenML to select the 602 supervised binary classification datasets on which at least 30 models have been evaluated with a 10-fold cross-validation. For each one, we randomly choose 30 models, fetch their predictions, and evaluate their performance with the metrics. We then compute the Spearman
model rank correlation matrix between the metrics. We also run the same experiment on the subset of 4 most imbalanced datasets. 

(insert figure here)

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

## References

Siblini, W., Fréry, J., He-Guelton, L., Oblé, F., & Wang, Y. Q. (2019). Master your Metrics with Calibration. arXiv preprint arXiv:1909.02827.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.

Flach, P., & Kull, M. (2015). Precision-recall-gain curves: PR analysis done right. In Advances in neural information processing systems (pp. 838-846).
