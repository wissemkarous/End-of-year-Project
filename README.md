# End-of-year-Project :

![image](https://github.com/wissemkarous/End-of-year-Project/assets/115191512/77550a2c-cd60-424c-896b-c70daa47e549)

TwoStreamLipNet. Here's a breakdown of its structure:

Convolutional Layers and Pooling:

Three 3D convolutional layers (conv1, conv2, conv3) with max-pooling (pool1, pool2, pool3) are used for feature extraction from the input data.
ReLU activation functions and 3D dropout are applied after each convolutional layer.
GRU Layers:

Two bidirectional Gated Recurrent Unit (GRU) layers (gru1, gru2) process the output from the convolutional layers in a sequential manner. GRUs are a type of recurrent neural network (RNN) that can capture temporal dependencies in the data.
Dropout is applied to the GRU outputs.
Lip Coordinates Processing:

A separate GRU layer (coord_gru) is employed to process lip coordinates.
Lip coordinates are reshaped and processed through the GRU, with dropout applied to the output.
Combination and Final Linear Layer:

The outputs from the two branches (GRU output and lip coordinates) are concatenated (torch.cat) and passed through a linear layer (FC) with a ReLU activation function.
The final output is permuted and returned.
Initialization:

The model parameters are initialized using the Kaiming normal initialization for convolutional and linear layers. For the GRU layers, a combination of uniform and orthogonal initializations is applied.
Forward Method:

The forward method defines the forward pass of the network. It takes two inputs, x (presumably some form of spatiotemporal data) and coords (lip coordinates), and processes them through the defined layers.
Overall, this model is designed for tasks involving spatiotemporal data, utilizing both 3D convolutional layers and recurrent layers to capture relevant features.
