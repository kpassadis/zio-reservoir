## An introduction to reservoir computing & Echo State Networks

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
At this point one may wonder: <i> why does this work? How is it possible to use randomly initialized stuff and still get meaningful predictions from the model?</i>

It turns out that the well known principle of projecting data to high dimensional spaces applies here. The information received by the readout signal is encoded into a very high non linear dimensional space.

But there is a catch. For this to work it is necessary that the reservoir system has certain properties. The most important is called the **Echo State Property**. As already mentioned, a reservoir is a dynamical system with memory and in the context of temporal processing this means that the system "remembers" past signals. But the past signals should not be as important as the more recents ones. The reservoir is initialized in a way to satisfy this <i> fading memory </i> property, also known as echo state property.

There exists a very common analogy to explain the physics of fading memory. Imagine standing at the edge of a cliff and shouting out very loudly. You will hear the echo of your voice. If you shout again you will hear the echo of the new shout as the previous echo gradually fades away. The reservoir behaves exactly like this canyon. When a continuous time-series signal (like electricity load or temperature changes) hits the network, it "shouts" into the reservoir. The internal state creates an echo of that information. As the next data point arrives a time step later, its fresh echo mixes with the bouncing, decaying audio waves of the previous step. The readout layer stands at the back of the canyon, listening to this complex acoustic blend of recent and past sounds to figure out what happens next.

If the canyon walls reflected sound perfectly without losing any energy, the echoes would never die down. A shout from three weeks ago would still be bouncing around at the exact same volume, creating a deafening wall of white noise that would completely drown out what you are saying right now.

