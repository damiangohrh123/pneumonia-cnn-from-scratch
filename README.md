# pneumonia-cnn-from-scratch

## 1. Introduction
This project focuses on building a Convolutional Neural Network (CNN) framework from scratch using only Python and NumPy. While standard neural networks are good for simple tasks, they struggle with images because they treat every pixel as an independent piece of data. This project aims to show how a CNN preserves the spatial structure of an image, showing how it understands how pixels relate to their neighbors and how it identify patterns like pneumonia in chest X-rays.

By building the system manually without libraries like TensorFlow or PyTorch, the project highlights the actual math behind how a computer "sees." Instead of relying on pre-built functions, every step from the sliding window of a convolution to the way error signals travel backward is coded from scratch. This approach ensures a complete understanding of the learning process rather than just treating the model as a black box.

### 1.1 Moving from Pixels to Patterns
In a basic neural network, images are flattened into a single line of numbers, which destroys the "shape" of the data. A CNN is designed specifically to handle 2D data by using filters. The main ideas behind this architecture include:

* **Local Receptive Fields:** Each part of the network only focuses on a small section of the image at a time, similar to how a human eye scans a page.

* **Weight Sharing:** The same filter is used across the entire image to look for specific features. This makes the model much more efficient and easier to train on a CPU.

* **Spatial Awareness:** The network can recognize a feature (like a specific lung texture) regardless of where it appears in the frame.

### 1.2 Medical Application: Pneumonia Detection
Medical images are often noisy and the differences between "Normal" and "Pneumonia" can be very subtle. A key part of this project is not just getting a high accuracy score, but making sure the network is reliable. In a medical setting, missing a sick patient is a much bigger error than a false alarm. Therefore, the project explores how to adjust the training process to prioritize finding every positive case.

## 2. System Architecture
A Convolutional Neural Network (CNN) is structured as a series of specialized layers that process data in a "spatial" way. Unlike standard networks that see an image as a flat list, a CNN treats it as a 3D volume with height, width, and depth. The architecture is designed to act as a funnel, taking a high-resolution input and gradually condensing it into high-level features that represent the objects or patterns within the image.

### 2.1 Input and Initial Processing
The architecture begins with the Input Layer, which stores the raw pixel data. In a CNN, this is usually represented as a 3D tensor where the dimensions correspond to the height, width, and the number of color channels (such as 1 for grayscale or 3 for RGB). Because the network needs to maintain the geometric relationship between pixels, this layer does not flatten the data. Instead, it preserves the grid structure so that the following layers can look for patterns in specific areas of the image.

### 2.2 Feature Extraction through Convolutional Layers
The Convolutional Layer is the primary engine of the network. It uses a set of learnable filters, also known as kernels, which are small matrices that slide across the input data. At every position, the filter performs a mathematical operation to see how well its own pattern matches the pixels in that specific spot. This process, illustrated in Fig. 1, allows the network to create "feature maps" that highlight where certain shapes, like edges or textures, are located. Because the same filter is used for the entire image, the network can recognize a pattern no matter where it appears, which makes it much more efficient than a standard fully connected layer.

<div align="center">
    <img src="images/sliding_window.png" width="600">
    <p align="center"><small><strong>Fig. 1.</strong> Sliding window (kernel) convolution operation and feature map generation. [1]</small><p>
</div>

### 2.3 Dimensionality Reduction via Pooling
To prevent the network from becoming too computationally heavy and to make it more reliable, Pooling Layers are placed between convolutional stages. These layers serve to "downsample" the feature maps, effectively shrinking the height and width of the data. Max-Pooling is the most common version, where the layer looks at a small window of pixels and only passes the highest value to the next stage, as demonstrated in Fig. 2. This ignores the exact location of a feature in favor of its general presence, which helps the network handle images where the subject might be slightly tilted or shifted.

<div align="center">
    <img src="images/max_pooling_example.png" width="400">
    <p align="center"><small><strong>Fig. 2.</strong> Max-pooling operation reducing the spatial resolution of a feature map by selecting maximum local values [1].</small></p>
</div>

### 2.4 The Fully Connected Head and Classification
Once the convolutional and pooling layers have extracted the most important visual information, the data must be converted into a format that can be used for a final decision. The 3D feature maps are "flattened" into a 1D vector and passed into a Fully Connected Layer. This part of the network acts like a standard classifier, looking at the entire collection of detected features to determine which category the image belongs to. In a binary system, a single output neuron with an activation function like Sigmoid is used to calculate the final probability of the target class.

<div align="center">
    <img src="images/system_architecture.png" width="800">
    <p align="center"><small><strong>Fig. 3.</strong> Complete CNN architecture showing the transition from 3D feature maps to a flattened 1D vector for classification. [2]</small><p>
</div>

## References
[1] J. Starmer, "Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)," YouTube, Jan. 14, 2020. [Online]. Available: https://www.youtube.com/watch?v=HGwqe6z1phM.

[2] Dharmaraj, "Convolutional Neural Networks (CNN) — Architecture Explained," Medium, [Online]. Available: https://owl.purdue.edu/owl/general_writing/grammar/using_articles.html.