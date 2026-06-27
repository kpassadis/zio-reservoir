

## An introduction to rezervoir computing

Reservoir computing is a machine learning framework focused on efficiently processing temporal patterns and sequential data. Models built using this paradigm can be considered a form of Recurrent Neural Networks (RNNs).

The simplest form of a neural network model based on reservoir consists of an input layer, a reservoir layer, and an output layer, called the readout.

The input signal is used to stimulate a high-dimensional, non-linear dynamical system, the reservoir layer. The reservoir, as any dynamical system that respects itself, maintains some internal state. When the input signal is received this is processed by the reservoir, the state is updated and passed through the readout layer which produces an output signal.




How do we devise a methodology for automatic tuning of an arbitrary architecture?

We need to modify the construction process so that all the layers that have hyperparameters (that is, all except the input layer)
maybe constructed as blueprints.
Then, during the tuning process, the initializers will suggest perturbations during the tuning process. So, the architecture will
be fixed but given the architecture the tuning process should be completely automatic.

Than architecting multiple blueprints 


## TODO

- [ ] Add the capability to specify multiple Readout layers.
- [ ] Add the capability to specify feedback connections by introducing time delay.

1. modify the forward method from 

```scala
def forward(x:DenseMatrix[Double], mode:Mode):DenseMatrix[Double]
```

to 

```scala
def forward(x:Batch):Batch
```

2. In the forward method of the Readout if the mode is Train add an additional column in the y part of the batch that will contain the target values.
Then, when forward is called if the current layer is calling it is a timedelay (or whatever it is called) it will check the model mode and 
pick up the ytrain or ytarget. How will it know whether the mode is training or eavluation? From the type of the batch. If it is of tytpe Labeled then
it must be operating in training mode. If it is operating in inference mode then it will receive an unlabeled batch. Because Batch serves as an end-to-end context container, Sequential simply maps a function composition over whatever enum branch is traveling through it.


## Metrics

The following metrics are available for evaluation

<u> Mean Absolute Percentage Error </u>

This returns a value between 0 and 100 which represents the absolute deviation of the predicted value from the target values as a fraction of the target value.

$$\text{MAPE} = \frac{100\%}{n \cdot m} \sum_{i=1}^n \sum_{j=1}^m \left| \frac{y_{ij} - \hat{y}_{ij}}{y_{ij}} \right|$$

<u> Normalized Mean Squared Error </u>

The percentage of variance explained by the model. The numerator is the mean squared error and the denominator the variance. 

$$\text{NMSE} = \frac{\sum (Y - \hat{Y})^2}{\sum (Y - \bar{Y})^2} = \frac{\text{MSE}}{\text{Variance}(Y)}$$

<u> Root Mean Squared Error </u>

While NMSE gives you a relative ratio, RMSE gives you an error score in the exact same units as your target data. If you are predicting a power in MW, the RMSE tells you your average error in MW.

$$\text{RMSE} = \sqrt{\frac{1}{n \cdot m} \sum_{i=1}^n \sum_{j=1}^m (y_{ij} - \hat{y}_{ij})^2}$$


Because it squares the errors before averaging them, RMSE gives a disproportionately high penalty to large errors. This is a critical metric if an occasional massive prediction spike would ruin your downstream application.

<u> Mean Absolute Error </u>

MAE measures the average magnitude of the errors without considering their direction. Unlike RMSE, it weights all errors linearly.

$$\text{MAE} = \frac{1}{n \cdot m} \sum_{i=1}^n \sum_{j=1}^m |y_{ij} - \hat{y}_{ij}|$$

Comparing RMSE and MAE can provide insights into your model's error distribution:

If $\text{RMSE} \approx \text{MAE}$, the model makes small, consistent errors across the entire timeline.
If $\text{RMSE} \gg \text{MAE}$, the model is highly accurate most of the time but suffers from a few catastrophic, massive misses.

<u> Direction Accuracy / Sign Test <u/>

In time-series, getting the exact numeric value right is often less important than predicting whether the signal will go UP or DOWN on the next step. Direction Accuracy measures the percentage of timesteps where the model correctly guessed the sign of the trend.

$$\text{DA} = \frac{1}{N-1} \sum_{t=1}^{N-1} \mathbb{I}\left( \text{sign}(y_{t+1} - y_t) == \text{sign}(\hat{y}_{t+1} - y_t) \right)$$

An model can sometimes achieve an amazing (low) MSE by simply predicting a smoothed, lagging version of yesterday's value. However, a lagging model will have a terrible Direction Accuracy ($< 50\%$). This metric acts as ta defense against a model that has simply learned to cheat by copying the past.

