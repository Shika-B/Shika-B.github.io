+++
title = "One-shot tabular prediction"
date = "2026-05-27T13:44:57+01:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["Computer Science", "Machine Learning", "Tabular Prediction", "Foundation Models"]
+++

Most of the data that actually runs the world lives in spreadsheets. Hospital records, bank transactions, sensor logs, 
messy CSV your colleague emailed you last Tuesday: it's really rows and columns all the way down. While we have been 
hearing quite a lot about language models writing poetry and code and image models painting cathedrals, the 
humble tabular prediction task was left to older methods[^1] for quite a long time. That changed a few years ago, with 
the introduction of various tabular foundation models[^2].

Consider the following table, where unknown entries are marked by red question 
marks:
| Cloud cover (%) | Humidity (%) | Pressure (hPa) |                      Wind (km/h) | Season |          Rain tomorrow           |
| --------------: | -----------: | -------------: | -------------------------------: | :----: | :------------------------------: |
|              20 |           45 |           1021 |                                8 | summer |                no                |
|              85 |           88 |            998 |                               22 | autumn |               yes                |
|              60 |           72 |           1009 |                               12 | spring |                no                |
|              90 |           80 |           1002 | <span style="color:red">?</span> | winter |               yes                |
|              40 |           55 |           1015 |                               10 | summer |                no                |
|              75 |           90 |           1005 |                                6 | spring |               yes                |
|              30 |           65 |           1018 |                               15 | autumn |                no                |
|              80 |           78 |           1007 |                               14 | winter | <span style="color:red">?</span> |

On tables like this one, the task of filling the red question marks with reasonable guesses is a *tabular prediction task*.
One wishes to fill the remaining entries by choosing the most probable value given what's known about the row (i.e. the 
other known entries in the same row). It's winter, and it's cloudy, so I guess it will rain. I know that because I 
have an idea of what the fields mean and how they correlate. That's better than random guessing, but it's probably even
better to also lean on the rows that are fully known, so as to learn the general relations and correlations between the
columns. This is pretty much the Bayesian approach. Unsurprisingly, the Bayesian approach is the standard 
framework for tabular prediction.

Before introducing modern methods, it's worth spending a minute on the two methods that have been winning at this game 
for years. Both are built out of *decision trees*.

## The binning trick
One small but useful trick before we start. From here on we'll pretend every target is categorical, even when it's 
really a continuous number. Predicting a real-valued $y_\star$ (wind speed, say) gets turned into a classification 
problem by *binning*: chop the range of $y$ into $B$ buckets $[b_0, b_1), [b_1, b_2), \dots, [b_{B-1}, b_B]$ and replace 
each true value with the index of the bucket it falls into. The model then predicts a distribution over $B$ classes 
instead of a single scalar. You lose a bit of resolution (you're capped by the bin width), but you can work with a 
single uniform output head for your model. 

## Decision trees

A decision tree is a little flowchart. At each internal node, it asks a yes/no question about one feature, for instance 
"is season = winter" or "is humidity > 70%?", and routes the row left or right. At the leaves, it spits out a 
prediction: a class label if we're trying to predict a categorical value.

To build a decision tree that fits your data, you must train it. A common method is to greedily pick, at each node, the
feature and threshold that best separates the data. "Best" is measured by some impurity criterion. For classification, 
the most common one (Gini impurity) evaluates the probability that two rows reaching this node have different class 
labels. Formally, if the node "contains" class proportions $p_1, \dots, p_K$ then the Gini impurity of the node is 
defined as

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

A split is good if it produces child nodes with lower (weighted) impurity than the parent. You keep splitting until some
stopping rule kicks in — minimum samples per leaf, maximum depth, etc.

For categorical values, code would look something like this:

```python
def gini(y):
    # 1 - sum_k p_k^2 over the classes present in y
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - sum(p ** 2)

def split_score(y_left, y_right):
    # weighted impurity of the two children
    n = len(y_left) + len(y_right)
    return (len(y_left)  / n) * gini(y_left) \
         + (len(y_right) / n) * gini(y_right)

def candidate_thresholds(column):
    # midpoints between consecutive unique values:
    # any t between v_i and v_{i+1} produces the same split,
    # so the midpoint is the canonical representative.
    v = sorted(set(column))
    return [(v[i] + v[i + 1]) / 2 for i in range(len(v) - 1)]

def best_split(X, y):
    best = (None, None, +inf)   # (feature, threshold, score)
    for j in range(num_features(X)):
        for t in candidate_thresholds(X[:, j]):
            left  = X[:, j] <= t
            right = X[:, j] >  t
            s = split_score(y[left], y[right])
            if s < best[2]:
                best = (j, t, s)
    return best[0], best[1]

def fit_decision_tree(X, y, depth=0, max_depth=10, min_samples=2):
    if depth >= max_depth or len(y) < min_samples or gini(y) == 0:
        return Leaf(prediction=majority_class(y))
    j, t = best_split(X, y)
    if j is None:                       # no useful split exists
        return Leaf(prediction=majority_class(y))
    left, right = X[:, j] <= t, X[:, j] > t
    return Node(
        feature=j, threshold=t,
        left  = fit_decision_tree(X[left],  y[left],  depth + 1, max_depth, min_samples),
        right = fit_decision_tree(X[right], y[right], depth + 1, max_depth, min_samples),
    )
```

Once the decision tree is trained, it is straightforward to use it for inference on new rows where the label to 
predict is not directly known. A single tree is interpretable and fast, but also a very poor model: small changes to 
the data can produce significantly different trees, and deep trees overfit. The two methods below are both ways of 
taming this fragility by combining many trees.

## Random forests

The idea: grow a lot of trees, each on a random subset of the rows and a random subset of the columns, then average 
their predictions. Each tree is a noisy, biased guess on its own, but the noise tends to cancel out when you average[^3].

Formally, if $T_1, \dots, T_B$ are trees fit on samples of the data, the forest's prediction is

$$\hat{y}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

for regression, or a majority vote for classification.

```python
def predict_row(tree, x):
    # walk the tree from root to a leaf, branching on x[feature] <= threshold
    if isinstance(tree, Leaf):
        return tree.prediction
    if x[tree.feature] <= tree.threshold:
        return predict_row(tree.left, x)
    else:
        return predict_row(tree.right, x)

def predict_all(tree, X):
    # one prediction per row of X
    return np.array([predict_row(tree, x) for x in X])

def random_forest(X, y, B=500):
    trees = []
    for _ in range(B):
        rows = bootstrap_sample(X)             # sample with replacement
        cols = random_subset(features(X))      # pick a random feature subset
        trees.append(fit_decision_tree(X[rows, cols], y[rows]))
    return trees

def predict_forest(trees, x):
    return mean(predict_row(t, x) for t in trees)
```

The trees don't talk to each other, which allows for parallelisation in real life implementations. 

## Gradient boosting

Boosting plays a different game. Instead of growing trees independently, it grows them *sequentially*, where each new 
tree tries to fix the mistakes of the set of trees so far. If the current ensemble predicts $\hat{y}_t(x)$, the next 
tree $T_{t+1}$ is fit to the *residuals*: the difference between the true class and the predictions.

```python
def gradient_boost(X, y, T=500, eta=0.1): # eta is a tunable learning rate
    pred = full(len(y), mean(y)) # start with any bad prediction
    trees = []
    for _ in range(T):
        residuals = y - pred 
        tree = fit_decision_tree(X, residuals)  # tree learns to predict residuals
        pred = pred + eta * predict_all(tree, X)  # nudge each prediction toward y
        trees.append(tree)
    return trees
```

The catch is that this iterative correction can overfit if you let it run too long, so tuning $T$ and $\eta$ matters. 
The payoff is that boosted trees are very hard to beat on tabular data.

## One-shot prediction

The biggest downside of these methods is that they require careful finetuning of hyperparameters, and training. This 
turns out to be a big bottleneck in real life, and people work around it by letting humongous grid searches run for days
to optimize these hyperparameters. And the actual training in itself also takes a lot of time. 

If only we could just *show* the model a table and have it tell us what's in the missing cells. After all,
LLMs don't get retrained every time you hand them a new document. They were pretrained once, on enormous amounts of 
text, and from that they learned something general enough that "what's the next token in this paragraph" can be answered
in one forward pass.

The hope is that the same trick works for tables: pretrain a model on millions of synthetic (or real) tables of all 
shapes and sizes, and it learns the meta-task of "given a few rows, predict the missing cell" well enough that, 
at inference time, you just feed it your CSV and read off the answer. Hence the name *one-shot prediction*.

The rest of this post is about some of the ideas that actually make this possible.

## Pretraining on a synthetic prior

The main trick that makes this work, popularized by TabPFN[^4], is to **generate the training data instead of collecting 
it**. Collecting billions of real tables with reliable labels to learn on is basically impossible. Instead, they imagine
a prior $p(\text{table})$ over all the kinds of tables you might ever encounter: how many rows, how many columns, what 
the relationships between columns look like, how much noise sits on top. If you can sample from that prior, you can 
produce an infinite stream of synthetic tables, each with a known "ground-truth" missing cell, because you generated it
yourself.

Crucially, tables are *not* random matrices. The goal is to sample tables that have meaningful relations between 
columns, so that some kind of Bayesian prediction is possible, since sheets in real life do have those kinds of 
relations. Predicting values that are uniformly distributed is both impossible and useless anyway.[^5]  

In practice, the prior is built out of small random Bayesian networks. You sample a Directed Acyclic Graph (DAG), sample
functional relationships along the edges (often parameterized by tiny neural networks), push noise through, and read off
a table. The schema looks roughly like:

```python
def sample_table():
    n_features = randint(1, 100)
    n_rows     = randint(20, 1000)
    graph      = sample_random_dag(n_features + 1)        # +1 for the target
    functions  = {edge: sample_random_function() for edge in graph.edges}
    noise      = sample_noise_levels(graph.nodes)
    X, y = simulate(graph, functions, noise, n_rows)
    return X, y
```

Each call gives you a new synthetic dataset, complete with the right answer, without ever needing a human to curate it.

## The transformer as a posterior approximator

Now you take a transformer and train it as you would with an LLM, but instead of "predict the next token", the input 
is a whole table: a *context* of labeled rows $(x_1, y_1), \dots, (x_n, y_n)$ plus a *query* row $x_\star$ whose 
label $y_\star$ is hidden. The output is a distribution over $y_\star$.

The loss is the cross-entropy of that distribution against the true label you generated. 
In [Müller et al.](https://arxiv.org/abs/2112.10510) they prove something that feels fairly intuitive: optimizing for 
this loss is exactly optimizing for the posterior prediction distribution, i.e. when you know both about the prior and 
the context.

The transformer isn't being told "here's a dataset, fit it". The prior lives in the weights and only the posterior 
update happens in the attention layers, in a single forward pass.

For instance, for our weather table, the six full rows go in as context and each row containing a red `?` goes in as a
separate[^6] query. The model spits out a probability over `{yes, no}` for the rain column and a distribution over wind
speeds for the other, and you're done.

## Where this breaks

The pretty story has limits. Current tabular foundation models do best on small-to-medium tables with up to a few 
thousand rows and a hundred or so columns, the regime the prior was trained to cover and the only regime the 
architecture is capable of handling right now. Pushing them to a million-row dataset with hundreds of categorical 
features is ongoing work (TabPFN v3 apparently managed to reach this milestone but I haven't verified it).

The main bottleneck is memory: attention is quadratic in the context length, and the context *is* the dataset. 
There's also the hard question of designing synthetic priors that properly cover the weirdness that real tables 
exhibit. Older methods still beat foundation models on Kaggle competitions for instance, often comfortably.

Still, the idea that one day we may have a strong and general foundation model for this kind of task is really 
exciting.

For anyone interested, there is also a pedagogical paper (with associated code that can be run on a laptop) that goes
into much more detail on the technical side of how Tabular Foundation Models (and more specifically TabPFN-family 
models) work, called [nanoTabPFN](https://arxiv.org/pdf/2511.03634). The code is available
[here](https://github.com/automl/nanoTabPFN).

# Comments

Sorry, I'm too lazy to load a proper comment system plugin: see the associated [github issue](https://github.com/Shika-B/Shika-B.github.io/issues/3).


[^1]: Maybe because Random forests and gradient boosting methods are still extremely strong baselines.
[^2]: *Foundation model* is a general term that basically means a model that can do a large variety of tasks because 
it has been trained on a lot of diverse data. Promptable models, both for text and image generation, typically lie in 
this category. 
[^3]: Something something Central Limit Theorem.
[^4]: TabPFN (Hollmann et al., [2022](https://arxiv.org/abs/2207.01848)) is the canonical example. The same authors 
later released a much-improved TabPFN v2 in *Nature*. Recently, they announced TabPFN v3 [here](https://arxiv.org/pdf/2605.13986). 
[^5]: Except for winning the lottery I guess.
[^6]: In practice this is batched on GPU, of course.