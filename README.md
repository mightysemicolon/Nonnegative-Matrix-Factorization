# Document Clustering with Nonnegative Matrix Factorization

Clustering text documents with NMF and comparing it against k-means, on two standard corpora (TDT2 and Reuters-21578).

The idea: factorize the term-document matrix as `A ≈ W H` with nonnegative `W` and `H` and rank `k` = number of clusters. Each column of `W` is a "topic" and column `j` of `H` says how much document `j` uses each topic. The document is assigned to the topic with the largest entry in its column of `H`.

## Data

Both datasets are given as document-term matrices with category labels (`fea` and `gnd` in the `.mat` files).

| Dataset | Documents | Terms | Categories |
|---|---|---|---|
| TDT2 | 9,394 | 36,771 | 30 |
| Reuters-21578 | 8,293 | 18,933 | 65 |

I made four subsets:

- **TDT2 top 10**: the 10 largest categories (7,456 documents)
- **TDT2 last 10**: the 10 smallest categories (653 documents)
- **Reuters top 10**: the 10 largest categories. The first two are much bigger than the others (3,713 and 2,055 documents), so I sampled 5% of the first, 10% of the second and 40% of the other eight. About 1,000 documents.
- **Reuters top 20**: same sampling for the first two categories, all documents of the next 18. About 2,400 documents.

## Methods

- **NMF**: scikit-learn `NMF` with `nndsvd` initialization, label = argmax over the column of `H`.
- **k-means**: scikit-learn `KMeans` with random initialization, on the same matrices.

## Evaluation

Three measures, all computed against the true categories:

- **Rand index** (`rand_score` from scikit-learn)
- **Purity**: for each cluster take the count of its most common true category, sum over clusters, divide by number of documents. Higher is better.
- **Entropy**: how mixed the true categories are inside each cluster, normalized by `log(k)`. Lower is better.

Purity and entropy are implemented by hand from the confusion matrix.

## Results

| Dataset | NMF Rand | NMF Purity | NMF Entropy | k-means Rand | k-means Purity | k-means Entropy |
|---|---|---|---|---|---|---|
| TDT2 top 10 | 0.902 | 0.838 | 0.237 | 0.630 | 0.554 | 0.538 |
| TDT2 last 10 | 0.947 | 0.824 | 0.233 | 0.681 | 0.538 | 0.532 |
| Reuters top 10 | 0.879 | 0.680 | 0.411 | 0.563 | 0.462 | 0.691 |
| Reuters top 20 | 0.911 | 0.609 | 0.418 | 0.674 | 0.373 | 0.651 |

NMF is clearly better than k-means on all four subsets and on all three measures. Reuters is harder than TDT2 for both methods.

The Reuters subsets are sampled randomly, so these numbers change a little from run to run.

At the end of the notebook the documents are projected to 2D with PCA and t-SNE and colored by the predicted cluster, for both methods.

## Files

- `Project-6.ipynb` : the notebook
- `TDT2.mat`, `Reuters21578.mat` : the datasets

## Requirements

numpy, scipy, pandas, scikit-learn, matplotlib
