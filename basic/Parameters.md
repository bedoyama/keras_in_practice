Parameters

1. Dropout (0.3)
2. # of Epochs: improves model until reaches a point where only marginal gains are provided at a high computational cost
3. MNIST_LR -> Learning Rate (0.001)
4. NHIDDEN (256): improves model until reaches a point where only marginal gains are provided at a high computational cost
5. BATCH_SIZE (128) for stochastic gradient descent

Other functions

1. model.predict(x) - used to predict new labels
2. model.evaluate() - used to compute loss values
3. model.predict_classes() - used to compute category outputs
3. model.predict_proba() - used to compute class probabilities