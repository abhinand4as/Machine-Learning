 
 
 ## Normalizing Feature Values
 

* we'd like our features to have reasonable scales
  *  Roughly zero centered,[-1,1] range often works well
  *  Helps gradient descent converge;avoid NaN trap
  *  Avoiding outlier values can also help
  
  
*  Can use a few standard methods:
    *  Linear scaling
    *  Hard cap
    *  Log scaling

### Linear Scaling

It can be a good standard practice to normalize the inputs to fall within the range -1, 1. 
This helps SGD not get stuck taking steps that are too large in one dimension, or too small in another. 

```javascript
def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)
```


```javascript
 processed_features = pd.DataFrame()
  processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
  processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
  processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
```

**Different Scaling functions:**

```javascript
def log_normalize(series):
  return series.apply(lambda x:math.log(x+1.0))

def clip(series, clip_to_min, clip_to_max):
  return series.apply(lambda x:(
    min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
  return series.apply(lambda x:(1 if x > threshold else 0))
```


## Optimizers

* Gradient Descent
* Adagrad
* Adam

The Adagrad optimizer is one alternative. The key insight of Adagrad is that it modifies the learning
rate adaptively for each coefficient in a model, monotonically lowering the effective learning rate. 
This works great for convex problems, but isn't always ideal for the non-convex problem Neural Net training.
You can use Adagrad by specifying AdagradOptimizer instead of GradientDescentOptimizer. 
Note that you may need to use a **larger learning rate with Adagrad**.

For non-convex optimization problems, Adam is sometimes more efficient than Adagrad. To use Adam, 
invoke the tf.train.AdamOptimizer method. This method takes several optional hyperparameters as arguments, 
but our solution only specifies one of these (learning_rate). In a production setting, you should specify and tune 
the optional hyperparameters carefully.

Adagrad optimizer

```javascript
_, adagrad_training_losses, adagrad_validation_losses = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.5),
    steps=500,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
 ```
 
 ### Graph of loss metrics of different optimizers
 
 ```javascript
 plt.ylabel("RMSE")
plt.xlabel("Periods")
plt.title("Root Mean Squared Error vs. Periods")
plt.plot(adagrad_training_losses, label='Adagrad training')
plt.plot(adagrad_validation_losses, label='Adagrad validation')
plt.plot(adam_training_losses, label='Adam training')
plt.plot(adam_validation_losses, label='Adam validation')
plt.plot(g_training_losses, label='Gradient Descent training')
plt.plot(g_validation_losses, label='Gradient Descent Validation')
_ = plt.legend()
```
