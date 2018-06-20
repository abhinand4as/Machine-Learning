# L1 Regularization

One way to reduce complexity is to use a regularization function that encourages weights to be exactly zero. 
For linear models such as regression, a zero weight is equivalent to not using the corresponding feature at all. 
In addition to avoiding overfitting, the resulting model will be more efficient.
L1 regularization is a good way to increase sparsity.

**L1 regularization reduces the model size**

* ftrl optimizer is best with L1 regularization
* feature crossing is applied

## Calculation model size :

```javascript
def model_size(estimator):
  variables = estimator.get_variable_names()
  size = 0
  for variable in variables:
    if not any(x in variable 
               for x in ['global_step',
                         'centered_bias_weight',
                         'bias_weight',
                         'Ftrl']
              ):
      size += np.count_nonzero(estimator.get_variable_value(variable))
  return size
```

## Create a linear classifier object.


 ```javascript
 my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
 my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )
```

## Training with regularization rate 0.1

```javascript
linear_classifier = train_linear_classifier_model(
    learning_rate=0.1,
    regularization_strength=0.1,
    steps=300,
    batch_size=100,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
print("Model size:", model_size(linear_classifier))
```
