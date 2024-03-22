# End-of-year-Project :
### Introduction :
this  is an advanced neural network model designed for accurate lip reading by incorporating lip landmark coordinates as a supplementary input to the traditional image sequence input. This enhancement to the original LipReading  architecture aims to improve the precision of sentence predictions by providing additional geometric context to the model.
### Features:
Dual Input System: Utilizes both raw image sequences and corresponding lip landmark coordinates for improved context.<br>
Enhanced Spatial Resolution: Improved spatial analysis of lip movements through detailed landmark tracking.<br>
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
# Result : 
![image](https://github.com/wissemkarous/End-of-year-Project/assets/115191512/9e2d5755-af81-4c04-8746-63d9f26c858e)<br>
![image](https://github.com/wissemkarous/End-of-year-Project/assets/115191512/9d14249e-081a-4283-b7c5-ea8aa6d94ca4)<br>
![image](https://github.com/wissemkarous/End-of-year-Project/assets/115191512/2bf7dcbd-e0c7-454d-993a-2c0e92bbf000)
#### Here : <br>

![image](https://github.com/wissemkarous/End-of-year-Project/assets/115191512/3e3d5537-2570-408d-be27-476bade0e87e)

We achieved a lowest WER of 1.7%, CER of 0.6% and a loss of 0.0256 on the validation dataset.

## DEMO :
[Check The Link  ](https://huggingface.co/spaces/wissemkarous/PFA-Demo) <br>
Huggingface spaces🤗<br>
Author ©️ : Wissem Karous <br>
Made with ❤️

