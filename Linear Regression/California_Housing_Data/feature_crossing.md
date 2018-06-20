## Feature Crossing

   To solve the nonlinear problem, create a feature cross.
A feature cross is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together.
(The term crosscomes from cross product.) Let's create a feature cross named x3 by crossing x1 and x2 :

*x3 = x1 x2*

We treat this newly minted x3 feature cross just like any other feature. The linear formula becomes:

*y = b + w1 x1 + w2 x2 + w3 x3*

### FTRL Optimization Algorithm

Follow The Regularized Leader

  High dimensional linear models benefit from using a variant of gradient-based optimization called FTRL. 
This algorithm has the benefit of scaling the learning rate differently for different coefficients, 
which can be useful if some features rarely take non-zero values (it also is well suited to support L1 regularization). 
We can apply FTRL using the FtrlOptimizer.

### Bucketized (Binned) Features
  Bucketization is also known as binning.
We can bucketize population into the following 3 buckets (for instance):
* bucket_0 (< 5000): corresponding to less populated blocks
* bucket_1 (5000 - 25000): corresponding to mid populated blocks
* bucket_2 (> 25000): corresponding to highly populated blocks

Given the preceding bucket definitions, the following population vector:

[[10001], [42004], [2500], [18000]]

becomes the following bucketized feature vector:

[[1], [2], [0], [1]]

The feature values are now the bucket indices. Note that these indices are considered to be discrete features. 
Typically, these will be further converted in one-hot representations as above, but this is done transparently.
To define feature columns for bucketized features, instead of using numeric_column, we can use bucketized_column, 
which takes a numeric column as input and transforms it to a bucketized feature using the bucket boundaries specified 
in the boundardies argument. The following code defines bucketized feature columns for households and longitude; 
the get_quantile_based_boundaries function calculates boundaries based on quantiles, so that each bucket contains an
equal number of elements.

## Feature Crosses
  Crossing two (or more) features is a clever way to learn non-linear relations using a linear model. 
In our problem, if we just use the feature latitude for learning, the model might learn that city 
blocks at a particular latitude (or within a particular range of latitudes since we have bucketized it) 
are more likely to be expensive than others. 
Similarly for the feature longitude. However, if we cross longitude by latitude, the crossed feature
represents a well defined city block. If the model learns that certain city blocks (within range of latitudes and longitudes)
are more likely to be more expensive than others, it is a stronger signal than two features considered individually.

Currently, the feature columns API only supports **discrete features** for crosses. To cross two continuous values,
like latitude or longitude, we can **bucketize** them.
If we cross the latitude and longitude features (supposing, for example, that longitude was bucketized into 2 buckets, 
while latitude has 3 buckets), we actually get six crossed binary features. Each of these features will get its own separate
weight when we train the model.

``` javascript
long_x_lat = tf.feature_column.crossed_column(
  set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000) 
```

