## An introduction to rezervoir computing

Reservoir computing is a machine learning framework focused on efficiently processing temporal patterns and sequential data. Models built using this paradigm can be considered a form of Recurrent Neural Networks (RNNs).

The simplest form of a neural network model based on reservoir consists of an input layer, a reservoir layer, and an output layer, called the readout.

The input signal is used to stimulate a high-dimensional, non-linear dynamical system, the reservoir layer. The reservoir, as any dynamical system that respects itself, maintains some internal state. When the input signal is received this is processed by the reservoir, the state is updated and passed through the readout layer which produces an output signal.

The mathematical representation of this process is the following:

When a raw input vector $u(t)$ arrives, a constant bias term is appended. The reservoir updates its internal activation state vector $x(t)$ by blending its historical memory with a new non-linear projection:

$$\tilde{x}(t) = \tanh(W_{in} u_{bias}(t) + W_{res} x(t-1))$$

$$x(t) = (1 - \gamma) \cdot x(t-1) + \gamma \cdot \tilde{x}(t)$$

Where:
- $u_{bias}(t)$ is the input vector scaled by the input scaling factor and extended with a bias coefficient.
- $\gamma \in (0, 1]$ represents the leaking rate. This parameter acts as a temporal low-pass filter: a smaller leak rate slows down state progression to track long-term historical trends, while a larger leak rate forces the system to react instantly to sudden exogenous shifts.

The process described above is known in Neural Network terminology as the feedforward step. A new input is pushed into a neural network model, this is propagated through its internal structure and an output is generated. The same idea applies to a reservoir model. What is different, however, is the learning process. In a Neural Network during training after the output has been produced it is typically compared to some target value, a loss is calculated and the error gradients are propagated backwards through the structure and the parameters get updated. This is not the case in reservoir models. Instead of running a computationally expensive optimization across the entire network, the learning process in Reservoir Computing is decoupled from the hidden layers.

In Neural Networks built using reservoir layers (also known as **Echo State Networks**) the only trainable parameters are the weights of the readout layer. The reservoir layer consists of randomly generated weights which stay fixed.

Because of this unique setup, the training process avoids the standard backpropagation algorithm and its associated computational overhead. Instead, training an Echo State Network is equivalent to solving a linear regression problem. 
The key question is: <i> why does this work? </i>


