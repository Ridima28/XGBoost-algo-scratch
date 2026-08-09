# XGBoost From Scratch

A learning-oriented implementation of **XGBoost (Extreme Gradient Boosting)** from scratch using Python and NumPy/Pandas.

The goal of this project is not to reproduce every optimization present in the official XGBoost library, but to understand the **mathematical foundations and internal mechanics** of XGBoost by implementing its core components manually.

The implementation includes:

* Gradient and Hessian computation
* Second-order Taylor approximation
* Optimal leaf weight calculation
* Tree splitting using gain
* L2 regularization
* Minimum child weight constraint
* Gamma-based split regularization
* Row subsampling
* Recursive tree construction
* Boosting across multiple trees
* Comparison against the official `xgboost` implementation

---

## Table of Contents

* [Project Overview](#project-overview)
* [What is XGBoost?](#what-is-xgboost)
* [Gradient Boosting Intuition](#gradient-boosting-intuition)
* [Why Does XGBoost Use Hessians?](#why-does-xgboost-use-hessians)
* [Mathematical Foundation](#mathematical-foundation)
* [XGBoost Objective Function](#xgboost-objective-function)
* [Second-Order Taylor Approximation](#second-order-taylor-approximation)
* [Optimal Leaf Weight](#optimal-leaf-weight)
* [Split Gain](#split-gain)
* [Tree Construction](#tree-construction)
* [Important Hyperparameters](#important-hyperparameters)
* [Training Process](#training-process)
* [Prediction Process](#prediction-process)
* [Implementation Details](#implementation-details)
* [Dataset](#dataset)
* [Results](#results)
* [Scratch vs Official XGBoost](#scratch-vs-official-xgboost)
* [Why Are the Results Slightly Different?](#why-are-the-results-slightly-different)
* [Project Structure](#project-structure)
* [How to Run](#how-to-run)
* [Key Learnings](#key-learnings)
* [Limitations](#limitations)
* [References](#references)

---

# Project Overview

This project implements the core idea behind XGBoost without directly using the XGBoost training algorithm.

Instead of calling:

```python
xgb.train(...)
```

the model manually:

1. Initializes predictions.
2. Calculates gradients.
3. Calculates Hessians.
4. Builds a regression tree using gradients and Hessians.
5. Calculates the optimal value of every leaf.
6. Finds the best feature and threshold for splitting.
7. Updates predictions.
8. Repeats the process for multiple boosting rounds.

The implementation is then compared with the official XGBoost library on the same dataset.

---

# What is XGBoost?

**XGBoost = Extreme Gradient Boosting**

XGBoost is an optimized implementation of gradient boosted decision trees.

The basic idea is:

> Build many weak decision trees sequentially, where each new tree tries to correct the errors made by the previous trees.

The final prediction is the sum of the predictions made by all trees:

$$
\hat{y}_i =
\hat{y}*i^{(0)}
+
\eta \sum*{t=1}^{T} f_t(x_i)
$$

where:

* $\hat{y}_i$ = final prediction
* $\hat{y}_i^{(0)}$ = initial/base prediction
* $\eta$ = learning rate
* $T$ = number of trees
* $f_t$ = prediction made by tree $t$

---

# Gradient Boosting Intuition

Suppose the actual values are:

```text
Actual:
[10, 20, 30]
```

and the current model predicts:

```text
Prediction:
[8, 15, 35]
```

The model is making errors.

Instead of rebuilding the entire model from scratch, gradient boosting asks:

> "What kind of tree can I add to reduce the current loss?"

The new tree learns from the **gradient of the loss function**.

After adding the new tree:

$$
Prediction_{new}
================

Prediction_{old}
+
\eta \times TreePrediction
$$

This process is repeated many times.

---

# Why Does XGBoost Use Hessians?

Traditional gradient boosting primarily uses the **first derivative (gradient)**.

XGBoost goes one step further and uses both:

* First derivative → Gradient
* Second derivative → Hessian

The gradient tells us:

> In which direction should the prediction move?

The Hessian tells us:

> How quickly is the gradient/loss changing?

This allows XGBoost to use a **second-order approximation** of the loss function.

---

# Mathematical Foundation

Suppose the model's prediction for sample $i$ at boosting round $t$ is:

$$
\hat{y}_i^{(t)}
===============

\hat{y}_i^{(t-1)} + f_t(x_i)
$$

The objective function is:

$$
Obj =
\sum_{i=1}^{n}
l(y_i,\hat{y}*i)
+
\sum*{t=1}^{T}\Omega(f_t)
$$

where:

* $l$ = training loss
* $y_i$ = actual value
* $\hat{y}_i$ = prediction
* $\Omega(f_t)$ = regularization term for tree $t$

---

# Second-Order Taylor Approximation

At boosting round $t$, XGBoost approximates the loss using Taylor expansion.

For sample $i$:

$$
l(y_i,\hat{y}_i^{(t-1)} + f_t(x_i))
$$

is approximated as:

$$
l(y_i,\hat{y}_i^{(t-1)})
+
g_i f_t(x_i)
+
\frac{1}{2}h_i f_t(x_i)^2
$$

where:

$$
g_i =
\frac{\partial l(y_i,\hat{y}_i)}
{\partial \hat{y}_i}
$$

and:

$$
h_i =
\frac{\partial^2 l(y_i,\hat{y}_i)}
{\partial \hat{y}_i^2}
$$

So:

* $g_i$ = gradient
* $h_i$ = Hessian

The constant loss term can be ignored while optimizing the new tree.

Therefore, the important part becomes:

$$
\sum_i
\left[
g_i f_t(x_i)
+
\frac{1}{2}h_i f_t(x_i)^2
\right]
$$

---

# Squared Error Objective

For this project, the objective function is Mean Squared Error.

The loss is:

$$
L =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

The implementation uses:

```python
class SquaredErrorObjective():

    def loss(self, y, pred):
        return np.mean((y - pred) ** 2)

    def gradient(self, y, pred):
        return pred - y

    def hessian(self, y, pred):
        return np.ones(len(y))
```

Therefore:

$$
g_i = \hat{y}_i-y_i
$$

and:

$$
h_i = 1
$$

The Hessian is constant for squared error.

---

# XGBoost Objective Function

XGBoost represents each tree using leaf weights.

Suppose a tree has leaves:

$$
j = 1,2,...,T
$$

and each sample belongs to a leaf.

The regularized objective can be expressed as:

$$
Obj =
\sum_j
\left[
G_j w_j
+
\frac{1}{2}(H_j+\lambda)w_j^2
\right]
+
\gamma T
$$

where:

$$
G_j = \sum_{i \in I_j}g_i
$$

and:

$$
H_j = \sum_{i \in I_j}h_i
$$

Here:

* $G_j$ = sum of gradients in leaf $j$
* $H_j$ = sum of Hessians in leaf $j$
* $w_j$ = leaf weight
* $\lambda$ = L2 regularization
* $\gamma$ = penalty for creating a new leaf

---

# Optimal Leaf Weight

To find the optimal leaf value, differentiate the objective with respect to $w_j$:

$$
\frac{\partial Obj}{\partial w_j}
=================================

G_j+(H_j+\lambda)w_j
$$

Set derivative equal to zero:

$$
G_j+(H_j+\lambda)w_j=0
$$

Therefore:

$$
\boxed{
w_j =
-\frac{G_j}{H_j+\lambda}
}
$$

This is one of the most important formulas in XGBoost.

The implementation uses:

```python
self.value = (
    -g[idx].sum()
    / (h[idx].sum() + self.reg_lambda)
)
```

So each leaf gets its own optimal prediction value.

---

# Split Gain

After finding possible splits, we need to determine:

> Is splitting this node actually useful?

Suppose a node is divided into:

```text
             Parent
             /    \
          Left    Right
```

Let:

* $G_L$ = gradient sum of left child
* $H_L$ = Hessian sum of left child
* $G_R$ = gradient sum of right child
* $H_R$ = Hessian sum of right child
* $G$ = gradient sum of parent
* $H$ = Hessian sum of parent

The gain is:

$$
Gain =
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-------------------------

\frac{G^2}{H+\lambda}
\right]
-------

\frac{\gamma}{2}
$$

The implementation uses:

```python
gain = 0.5 * (
    (sum_g_left**2 / (sum_h_left + self.reg_lambda))
    +
    (sum_g_right**2 / (sum_h_right + self.reg_lambda))
    -
    (sum_g**2 / (sum_h + self.reg_lambda))
) - self.gamma / 2
```

The algorithm chooses the split with the **highest positive gain**.

---

# How Does Splitting Work?

For every feature:

### Step 1 — Select the feature

```python
x = self.X.values[self.idx, feature_idx]
```

Only the samples belonging to the current node are considered.

---

### Step 2 — Sort the feature

```python
sort_idx = np.argsort(x)

sort_x = x[sort_idx]
sort_g = g[sort_idx]
sort_h = h[sort_idx]
```

Example:

```text
Feature values:

20  50  30  10  40

After sorting:

10  20  30  40  50
```

---

### Step 3 — Move samples from right to left

Initially:

```text
Left | Right
-----|----------------
     | 10 20 30 40 50
```

After moving 10:

```text
Left | Right
10   | 20 30 40 50
```

After moving 20:

```text
Left | Right
10 20 | 30 40 50
```

And so on.

At each position, the algorithm calculates the gain.

---

### Step 4 — Find a threshold

If:

```text
x_i = 20
x_next = 30
```

then:

$$
threshold =
\frac{20+30}{2}
===============

25
$$

Therefore:

```text
x <= 25  -> Left
x > 25   -> Right
```

---

# Tree Construction

The tree is built recursively.

The process is:

```text
                  Root
                /      \
             Left      Right
             /  \       /  \
           LL   LR     RL   RR
```

At each node:

1. Check all features.
2. Find the best threshold.
3. Calculate gain.
4. Select the best split.
5. Create left and right children.
6. Repeat recursively.

The recursion stops when:

```text
max_depth == 0
```

or no valid/beneficial split is found.

---

# Important Hyperparameters

## `learning_rate`

Controls how much each new tree contributes.

```python
learning_rate = 0.1
```

Formula:

$$
Prediction_{new}
================

Prediction_{old}
+
\eta TreePrediction
$$

Smaller learning rate:

* slower learning
* usually requires more trees
* can improve generalization

---

## `max_depth`

Controls maximum tree depth.

```python
max_depth = 5
```

Higher depth:

* more complex trees
* can capture complex relationships
* greater risk of overfitting

Lower depth:

* simpler trees
* less computationally expensive
* potentially underfits

---

## `subsample`

Controls the fraction of training rows used by each tree.

```python
subsample = 0.8
```

If there are 10,000 samples:

$$
0.8 \times 10000 = 8000
$$

So approximately 8,000 samples are randomly selected for that tree.

Implementation:

```python
sample_idx = self.rng.choice(
    len(y),
    size=math.floor(self.subsample * len(y)),
    replace=False
)
```

This is called **row subsampling**.

It can help reduce overfitting.

---

## `colsample_bynode`

Controls how many features are considered at each node.

```python
colsample_bynode = 1.0
```

For example, if there are 10 features:

```text
colsample_bynode = 1.0
→ consider all 10 features

colsample_bynode = 0.5
→ consider approximately 5 features
```

This is different from `subsample`.

| Parameter          | Samples or Features? | Applied Where? |
| ------------------ | -------------------- | -------------- |
| `subsample`        | Samples              | Each tree      |
| `colsample_bynode` | Features             | Each node      |

---

## `reg_lambda`

L2 regularization.

```python
reg_lambda = 1.5
```

It appears in the leaf weight formula:

$$
w =
-\frac{G}{H+\lambda}
$$

Higher $\lambda$ makes leaf weights smaller and can reduce overfitting.

---

## `gamma`

Controls how much gain is required before splitting.

```python
gamma = 0.0
```

Conceptually:

```text
High gamma
    ↓
Harder to split
    ↓
Simpler trees
    ↓
Less overfitting
```

The gain calculation includes:

$$
-\frac{\gamma}{2}
$$

---

## `min_child_weight`

Controls the minimum Hessian weight required in a child.

```python
min_child_weight = 25
```

A candidate split is rejected if:

$$
H_L < min_child_weight
$$

or:

$$
H_R < min_child_weight
$$

This prevents extremely small/weak child nodes.

---

# Training Process

The complete boosting process is:

```text
                  Training Data
                       |
                       v
              Initial Prediction
                       |
                       v
              Calculate Gradient
                       |
                       v
              Calculate Hessian
                       |
                       v
                Build Tree
                       |
                       v
            Find Best Splits
                       |
                       v
             Calculate Leaf
                  Weights
                       |
                       v
              Tree Prediction
                       |
                       v
          Update Current Prediction
                       |
                       v
                Next Tree
                       |
                       v
                     ...
```

Mathematically:

$$
\hat{y}^{(0)} = base_score
$$

Then for each tree:

$$
g_i =
\frac{\partial L}{\partial \hat{y}_i}
$$

$$
h_i =
\frac{\partial^2 L}{\partial \hat{y}_i^2}
$$

Build:

$$
f_t(x)
$$

and update:

$$
\hat{y}^{(t)}
=============

\hat{y}^{(t-1)}
+
\eta f_t(x)
$$

---

# Training Implementation

The core training loop is:

```python
for i in range(n_estimators):

    gradients = objective.gradient(
        y,
        current_prediction
    )

    hessians = objective.hessian(
        y,
        current_prediction
    )

    if self.subsample == 1:
        sample_idx = None
    else:
        sample_idx = self.rng.choice(
            len(y),
            size=math.floor(
                self.subsample * len(y)
            ),
            replace=False
        )

    tree = TreeBooster(
        X,
        gradients,
        hessians,
        self.params,
        self.max_depth,
        sample_idx
    )

    current_prediction += (
        self.learning_rate * tree.predict(X)
    )

    self.models.append(tree)
```

---

# Prediction Process

After training, the model contains multiple trees:

```text
Tree 1
Tree 2
Tree 3
...
Tree 50
```

The final prediction is:

$$
\hat{y}
=======

base_score
+
\eta
\sum_{t=1}^{T}Tree_t(x)
$$

Implementation:

```python
def predict(self, X):

    return (
        self.base_prediction
        +
        self.learning_rate
        *
        np.sum(
            [tree.predict(X) for tree in self.models],
            axis=0
        )
    )
```

---

# How Does a Single Tree Predict?

For every row, the tree traverses from root to leaf.

For example:

```text
                Feature 2 <= 5?
                 /        \
               Yes         No
               /            \
        Feature 1 <= 10?    Leaf
          /       \
       Leaf       Leaf
```

The prediction algorithm repeatedly checks:

```python
row[self.split_feature_idx] <= self.threshold
```

and moves left or right.

When a leaf is reached:

```python
return self.value
```

---

# Dataset

The implementation was tested using the **California Housing dataset** from scikit-learn.

```python
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(
    as_frame=True,
    return_X_y=True
)
```

The dataset was divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=43
)
```

---

# Configuration

The following parameters were used:

```python
params = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bynode': 1.0,
    'reg_lambda': 1.5,
    'gamma': 0.0,
    'min_child_weight': 25,
    'base_score': 0.5,
    'tree_method': 'exact',
}

num_boost_round = 50
```

---

# Scratch vs Official XGBoost

The same dataset and approximately the same configuration were used for both implementations.

### From-Scratch Model

```python
model_scratch = XGBoostModel(
    params,
    random_seed=42
)

model_scratch.fit(
    X_train,
    y_train,
    SquaredErrorObjective(),
    num_boost_round
)
```

### Official XGBoost

```python
dtrain = xgb.DMatrix(
    X_train,
    label=y_train
)

model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round
)
```

---

# Results

The final Mean Squared Error was:

| Model                |                     MSE |
| -------------------- | ----------------------: |
| From-Scratch XGBoost |  **0.2434125759558149** |
| Official XGBoost     | **0.24323066635317615** |

The difference is extremely small:

$$
Difference
\approx
0.00018191
$$

This demonstrates that the implementation successfully captures the **core mathematical behavior of gradient-boosted decision trees using the XGBoost-style objective and splitting mechanism**.

---

# Example Predictions

For a few test samples:

```text
Actual:
[1.478, 2.192, 1.734, 1.996, 1.683]

Scratch:
approximately
[1.45, 2.05, 1.63, 1.87, 1.80]

Official XGBoost:
[1.4555, 2.0550, 1.6351, 1.8756, 1.7992]
```

The predictions are close, showing that the scratch implementation is learning a very similar function.

---

# Why Are the Results Slightly Different?

The scratch implementation and official XGBoost implementation are **not expected to produce exactly identical predictions**.

Several differences can contribute.

### 1. Official XGBoost is highly optimized

The official implementation contains many engineering and optimization details that are not reproduced here.

---

### 2. Feature subsampling

The scratch implementation exposes:

```python
colsample_bynode
```

but a simplified implementation may not reproduce the exact feature sampling behavior of the production implementation.

---

### 3. Row subsampling

The scratch implementation uses:

```python
np.random.default_rng(seed=42)
```

The official XGBoost library has its own random-number generation and sampling implementation.

Therefore, even with the same seed, the exact sampled rows may differ.

---

### 4. Tree construction details

The official library contains sophisticated implementations of:

* split enumeration
* numerical handling
* missing values
* histogram algorithms
* parallelization
* pruning
* numerical optimizations

This project focuses on understanding the fundamental algorithm rather than reproducing every production optimization.

---

# Important Implementation Concept

One of the most important parts of this project is understanding the relationship between:

```text
Gradient
   ↓
Hessian
   ↓
Leaf Weight
   ↓
Split Gain
   ↓
Best Feature + Threshold
   ↓
Tree
   ↓
Prediction Update
```

The entire XGBoost tree-building process can essentially be understood through these quantities.

---

# Core Formulas Summary

### Gradient

$$
g_i =
\frac{\partial l(y_i,\hat{y}_i)}
{\partial \hat{y}_i}
$$

For squared error:

$$
\boxed{g_i = \hat{y}_i-y_i}
$$

---

### Hessian

$$
h_i =
\frac{\partial^2 l(y_i,\hat{y}_i)}
{\partial \hat{y}_i^2}
$$

For squared error:

$$
\boxed{h_i=1}
$$

---

### Leaf Weight

$$
\boxed{
w_j =
-\frac{G_j}{H_j+\lambda}
}
$$

---

### Split Gain

$$
\boxed{
Gain =
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-------------------------

\frac{G^2}{H+\lambda}
\right]
-------

\frac{\gamma}{2}
}
$$

---

### Prediction Update

$$
\boxed{
\hat{y}^{(t)}
=============

\hat{y}^{(t-1)}
+
\eta f_t(x)
}
$$

---

### Final Prediction

$$
\boxed{
\hat{y}
=======

base_score
+
\eta
\sum_{t=1}^{T}f_t(x)
}
$$

---

# Project Structure

A possible project structure:

```text
xgboost-from-scratch/
│
├── README.md
├── xgboost_scratch.py
├── notebook.ipynb
├── requirements.txt
└── results/
    └── predictions.csv
```

---

# Requirements

Install the required libraries:

```bash
pip install numpy pandas scikit-learn xgboost
```

---

# How to Run

Clone the repository:

```bash
git clone <your-repository-url>
cd xgboost-from-scratch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook or Python script.

---

# Key Learnings

This project helped demonstrate that XGBoost is not simply:

> "Many decision trees combined together."

Its strength comes from combining several ideas:

### 1. Gradient Boosting

Each tree learns from the errors of the current model.

### 2. Second-Order Optimization

XGBoost uses both gradients and Hessians.

### 3. Regularization

L2 regularization and split penalties help control model complexity.

### 4. Optimal Leaf Weights

Each leaf gets an analytically calculated optimal value:

$$
w =
-\frac{G}{H+\lambda}
$$

### 5. Gain-Based Splitting

The tree chooses the feature and threshold that produce the largest improvement in the objective.

### 6. Sampling

Row and feature subsampling can improve generalization and reduce computational cost.

---

# Limitations

This implementation is designed primarily for **educational purposes**.

It does not attempt to reproduce all features and optimizations of the official XGBoost library.

For example, the implementation does not fully reproduce:

* Missing-value handling
* Sparse matrix optimizations
* Histogram-based tree construction
* Parallel tree construction
* GPU acceleration
* Distributed training
* Advanced categorical feature handling
* All XGBoost objective functions
* All production-level numerical optimizations

Therefore, this project should be viewed as:

> **A simplified implementation of the core XGBoost algorithm for learning and experimentation.**

---

# Conclusion

Implementing XGBoost from scratch provides a much deeper understanding of what happens inside a gradient boosting algorithm.

The key idea can be summarized as:

```text
Current Predictions
        ↓
   Calculate Gradient
        ↓
   Calculate Hessian
        ↓
 Find Best Tree Split
        ↓
 Calculate Optimal Leaf Weights
        ↓
     Build Tree
        ↓
 Update Predictions
        ↓
   Repeat N Times
        ↓
 Final Prediction
```

The scratch implementation achieved an MSE of approximately:

$$
\boxed{0.24341}
$$

while the official XGBoost implementation achieved:

$$
\boxed{0.24323}
$$

The close results provide a useful sanity check that the fundamental concepts—**second-order optimization, leaf weights, gain-based splitting, regularization, and boosting**—have been implemented correctly.

---

# References

* Chen, T. & Guestrin, C. — **XGBoost: A Scalable Tree Boosting System**
* XGBoost official documentation
* scikit-learn California Housing dataset documentation

---

## Final Takeaway

The most important formulas to remember are:

$$
\boxed{g_i = \frac{\partial l}{\partial \hat{y}_i}}
$$

$$
\boxed{h_i = \frac{\partial^2 l}{\partial \hat{y}_i^2}}
$$

$$
\boxed{
w_j = -\frac{G_j}{H_j+\lambda}
}
$$

$$
\boxed{
Gain =
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-------------------------

\frac{G^2}{H+\lambda}
\right]
-------

\frac{\gamma}{2}
}
$$

and finally:

$$
\boxed{
\hat{y}^{(t)}
=============

\hat{y}^{(t-1)}
+
\eta f_t(x)
}
$$

These equations form the mathematical core of the XGBoost implementation developed in this project.