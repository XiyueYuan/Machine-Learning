## Decision Tree

- a tree-structured model in which each internal node represents a decision based on a feature
- and each branch corresponds to the outcome of that decision

The construction process is as follows:
1. Feature selection: Choose the features with strong discriminative power.
2. Decision tree generation: Build the decision tree based on the selected features.
3. Pruning: Use pruning techniques to alleviate overfitting.

### Entropy

- Greater entropy is, higher the uncertainty in data, more information 
- Less entropy is, less the uncertainty in data

Entropy
$$
H(X) = -\sum_{i=1}^{n} p_i \log_2 p_i
$$
For example: 
1. DatasetA: {ABCDEFGH}
$$
H(X) = -\frac{1}{8} \times log_2 \frac{1}{8} \times 8
= 3
$$
2. DatasetB: {AAAABBCD}
$$
H(X) = (-\frac{1}{2} \times log_2 \frac{1}{2} \times 4) + (-\frac{1}{4} \times log_2 \frac{1}{4} \times 2)+ (-\frac{1}{8} \times log_2 \frac{1}{8} \times 2)
$$

```python
import math
math.log(8, 2)
# or
from math import log
log(8, 2)
```
### Information Gain 
- If splitting on this feature (once) leads to a __*clear reduction*__ in the dataset’s overall entropy
- then this feature is a good split point and should be used for the current partition
