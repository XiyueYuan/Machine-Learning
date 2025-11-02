### KNN Algorithm
##### K-Nearest Neighbors
1. *'k' meaning*
- k means the number of __nearest neighbors__ the algorithm looks at when making a prediction
- if k is too large, resulting in under-fitting
- if k is too small, resulting in over-fitting

- __Euclidean distance__ 
$$
d=\sqrt{\sum_{k=1}^{n}​{(x_{test} - x_{train})^ 2}}
$$
- __Manhattan distance__
$$
d = \sum_{k=1}^{n}|x_{test} - x_{train}|
$$

2. *algorithm*
- classification 
    - discrete
    - by vote
- regression 
    - continuous
    - average

3. Feature preprocessing 
    1. Normalization: to compress features with different scales into the same range (usually between 0 and 1)

    $x' = \frac{x - x_{min}}{max - min}$
