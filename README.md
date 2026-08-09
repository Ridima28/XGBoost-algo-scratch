# XGBoost From Scratch

A from-scratch implementation of the core ideas behind **XGBoost (Extreme Gradient Boosting)** using Python and NumPy/Pandas.

The goal of this project was to understand what happens inside XGBoost instead of treating it as a black-box library.

## 🚀 What I Implemented

* Gradient and Hessian-based boosting
* Regression trees
* Greedy split selection
* Gain-based split evaluation
* Optimal leaf weight calculation
* L2 regularization (`reg_lambda`)
* Minimum child weight (`min_child_weight`)
* Split penalty (`gamma`)
* Row subsampling (`subsample`)
* Learning rate / shrinkage
* Multiple boosting rounds
* Tree-based prediction

## 🧠 Core Theory

XGBoost builds trees **sequentially**. Each new tree tries to correct the errors made by the existing ensemble.

For every boosting round:

1. Start with the current predictions.
2. Calculate the **gradient** and **Hessian** of the loss.
3. Build a decision tree using these values.
4. Evaluate possible splits using their **gain**.
5. Calculate the optimal value for each leaf.
6. Add the new tree's predictions to the existing predictions using the learning rate.

### Gradient & Hessian

* **Gradient:** tells the direction in which the prediction should move.
* **Hessian:** represents the curvature/rate of change of the loss.

Using both allows XGBoost to perform a second-order approximation of the loss.

### Tree Splitting

For every feature:

* Sort the feature values.
* Consider possible thresholds between consecutive values.
* Calculate the gain of each possible split.
* Select the feature and threshold with the highest gain.

Splits can be restricted using:

* `min_child_weight`
* `gamma`
* `max_depth`

### Regularization

XGBoost uses regularization to reduce overfitting.

In this implementation:

* `reg_lambda` → L2 regularization on leaf weights
* `gamma` → minimum improvement required for a split
* `max_depth` → controls tree complexity
* `min_child_weight` → prevents very small/weak child nodes

### Subsampling

`subsample` randomly selects a fraction of training samples for each tree.

This adds randomness and can help reduce overfitting.

## 📊 Validation

The implementation was tested on the **California Housing dataset** and compared with the official XGBoost implementation.

Example result:

| Model                |     MSE |
| -------------------- | ------: |
| From-Scratch XGBoost | ~0.2434 |
| XGBoost Library      | ~0.2432 |

The close results validate that the core boosting and tree-building logic is working correctly.

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* XGBoost (for comparison)

## 📚 Key Learning

This project helped me understand the internal mechanics of gradient boosting, especially how **gradients, Hessians, tree splits, gain, regularization, and sequential model updates** work together to build XGBoost.

The main objective was not to reproduce every production-level optimization of XGBoost, but to understand and implement its **core algorithm from first principles**.
