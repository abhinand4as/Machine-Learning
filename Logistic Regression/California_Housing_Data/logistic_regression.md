# Logistic Regression

To use logistic regression, simply use LinearClassifier instead of LinearRegressor.

```javascript
linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
```

## LogLoss

The loss function for linear regression is squared loss(L2 Loss). 
The loss function for logistic regression is Log Loss,
which is defined as follows:

```javascript
 training_log_loss = metrics.log_loss(training_targets, training_probabilities)
 validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
 ```
 ## Accuracy and ROC Curve
 A few of the metrics useful for classification are the model accuracy, 
 the ROC curve and the area under the ROC curve (AUC)
 
 ```javascript
evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
```


**NOTE:** When running train() and predict() on a LinearClassifier model, 
you can access the real-valued predicted probabilities via the "probabilities" key in the returned dict—e.g., 
predictions["probabilities"]. 
Sklearn's log_loss function is handy for calculating LogLoss using these probabilities.
