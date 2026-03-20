# Convolutional Neural Network From Scratch (Pneumonia)

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

### 2.3 Dimensionality Reduction via Pooling
To prevent the network from becoming too computationally heavy and to make it more reliable, Pooling Layers are placed between convolutional stages. These layers serve to "downsample" the feature maps, effectively shrinking the height and width of the data. Max-Pooling is the most common version, where the layer looks at a small window of pixels and only passes the highest value to the next stage, as demonstrated in Fig. 2. This ignores the exact location of a feature in favor of its general presence, which helps the network handle images where the subject might be slightly tilted or shifted.

### 2.4 The Fully Connected Head and Classification
Once the convolutional and pooling layers have extracted the most important visual information, the data must be converted into a format that can be used for a final decision. The 3D feature maps are "flattened" into a 1D vector and passed into a Fully Connected Layer. This part of the network acts like a standard classifier, looking at the entire collection of detected features to determine which category the image belongs to. In a binary system, a single output neuron with an activation function like Sigmoid is used to calculate the final probability of the target class.

<div align="center">
    <img src="images/system_architecture.png" width="800">
    <p align="center"><strong>Fig. 1.</strong> Complete CNN architecture showing the transition from 3D feature maps to a flattened 1D vector for classification. [1]<p>
</div>

## 3. Forward Propagation
Forward propagation is the mathematical process where input data travels through the network to generate a prediction. In a Convolutional Neural Network, this involves moving from a high-resolution raw image to a set of abstract features, and finally to a probability. Instead of processing the entire image at once, the forward pass breaks the image down into local patterns, applying linear transformations and non-linear activations at each stage. This sequential flow ensures that the model can build a complex understanding of the chest X-ray, starting with simple edges and ending with diagnostic indicators.

### 3.1 The Convolutional Operation and Feature Extraction
The core of the forward pass begins with the convolutional operation. As a filter slides across the input image, it performs a element-wise multiplication and summation, also known technically as a cross-correlation. For each position $(i, j)$ in the output map, the operation is calculated as:

$$z_{i,j} = \sum_{m} \sum_{n} I_{i+m, j+n} \cdot K_{m,n} + b$$

To understand how the computer actually processes the image, the terms can be broken down as follows:

**The Input Image and the Kernel Weights ($I_{i+m, j+n} \cdot K_{m,n}$):** At each position $(i, j)$, a $3 \times 3$ window slides across the input image $I$, capturing a patch of pixel values $I_{i+m,j+n}$. Each pixel in this patch is multiplied by a corresponding weight $K_{m,n}$ from the kernel, which is a small $3 \times 3$ grid of learned values that determines which patterns the filter responds to. The double summation $\sum_m \sum_n$ then accumulates all nine products across the rows ($m$) and columns ($n$) of the kernel into a single scalar $z_{i,j}$.

<div align="center">
    <img src="images/sliding_window.png" width="700">
    <p align="center"><strong>Fig. 2.</strong> The kernel slides across the input image, multiplying each pixel in the 3×3 patch by its corresponding weight​. The nine products are summed and a bias is added, producing one value in the feature map. Here, the products sum to 1 and a bias of −2 yields a final activation of −1. [2]<p>
</div>

**The Bias ($b$):** A learned scalar constant $b$ is added to the weighted sum. This shifts the activation threshold, allowing the network to adjust how sensitive a filter is to a given feature independently of the input pixel values. In Fig. 2, a bias of $b = -2$ is added to the sum of $1$, yielding a final activation of $z_{i,j} = -1$.

### 3.2 Non-Linearity through ReLU
Once the feature maps are generated, they are passed through a Rectified Linear Unit (ReLU) activation function. The primary purpose of this step is to introduce non-linearity, which allows the network to learn relationships that aren't just simple linear combinations of pixels. The ReLU function is defined as $f(z) = \max(0, z)$, meaning it allows positive signals to pass through unchanged while effectively "turning off" any negative values. This helps the network focus on the most relevant features and prevents the mathematical instability that can occur in deeper networks.

<div align="center">
    <img src="images/feature_map_relu.png" width="700">
    <p align="center"><strong>Fig. 3.</strong> Feature map passing through ReLU activation function. [2]<p>
</div>

### 3.3 Spatial Downsampling with Max Pooling
Following activation, the feature maps undergo Max Pooling to reduce their spatial dimensions. The network slides a $2 \times 2$ window across the feature map and selects only the maximum value within that window to move forward to the next layer. This operation is critical for maintaining "translation invariance," which means the network can still recognize a pattern even if it is shifted slightly in the image. Furthermore, by shrinking the height and width of the data by half, the pooling step significantly reduces the computational load for the subsequent layers without losing the most prominent features.

<div align="center">
    <img src="images/max_pooling.png" width="500">
    <p align="center"><strong>Fig. 4.</strong> Max pooling: each 2×2 region collapses to its single highest value, halving the map size. [2]<p>
</div>

### 3.4 Flattening and the Sigmoid Prediction
The final stage of the forward pass involves converting the 3D feature volumes into a 1D vector through a process called flattening. This vector is then passed to a single output neuron in a fully connected layer. To convert the raw signal into a usable diagnosis, the network applies the Sigmoid activation function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This function maps any input value into a strict range between 0 and 1, which represents the probability of the presence of pneumonia. A value closer to 1 indicate the network is confident in a positive diagnosis, while a value closer to 0 indicates a healthy scan.

## 4. Loss Function
The loss function is the mathematical tool the network uses to measure the gap between its predictions and the actual truth. It provides a single scalar value that represents the total error of the model for a given set of images. The objective of the entire training process is to minimize this number through optimization. In a classification task like pneumonia detection, the loss function does not just look at whether the network was right or wrong, but also evaluates how confident it was in its answer, punishing confident but incorrect predictions more severely.

### 4.1 Binary Cross-Entropy (BCE)
For binary classification, the network utilizes the Binary Cross-Entropy loss function. This function is specifically designed to work with the Sigmoid output from the final layer, which provides a probability between 0 and 1. The BCE function compares this probability to the actual label, where 0 represents a healthy scan and 1 represents a pneumonia scan. If the network predicts a high probability for a positive case and the label is actually positive, the loss is low. However, if the network is very confident about the wrong answer (for example, predicting a $0.99$ probability for pneumonia when the patient is actually healthy) the BCE function produces an extremely high loss value. This high error signal is what triggers significant adjustments to the weights during backpropagation.The mathematical formula for the loss is calculated as:

$$
L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

* **$y$ (The True Label):** This is the ground truth, which is either 0 or 1.
* **$\hat{y}$ (The Prediction):** This is the probability output by the Sigmoid function.
* $\log(\hat{y})$: The logarithm ensures that as the prediction gets further from the truth, the loss increases exponentially.

To understand why the formula looks like $L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$, we have to look at the two possible scenarios for a patient:

* **Scenario 1: The Positive Case ($y = 1$):**  
When the chest X-ray contains pneumonia, the second term $(1 - y)$ becomes zero since $(1-1) = 0$, leaving only $L = -\log(\hat{y})$. The network is now solely focused on how close the prediction $\hat{y}$ is to $1$.

* **Scenario 2: The Negative Case ($y = 0$):**  
When the scan is healthy, the first term $y$ becomes zero since $0 \log(\hat{y}) = 0$, leaving $L = -\log(1 - \hat{y})$. The network now only evaluates how close the prediction is to $0$.

### 4.2 Handling Class Imbalance and Medical Priority
A significant challenge in medical imaging is that datasets are often imbalanced, frequently containing far more healthy scans than pneumonia scans. If the loss function treats every error the same, the model might learn to simply guess "healthy" every time to keep the total error low, which is dangerous in a clinical setting. To prevent this, a weighted version of the loss function can be implemented. By adding a penalty multiplier to the pneumonia class, the network is punished more for a False Negative (missing a sick patient) than for a False Positive (a false alarm).

The standard BCE formula is modified by introducing a weight factor, $w_{pos}$, to the positive $(y=1)$ term:

$$
L = -[w_{pos} \cdot y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

In a hospital, a "False Alarm" results in an unnecessary follow-up test, but a "Missed Case" results in a patient sent home without treatment. If we set $w_{pos} = 5$, the "pain" or error signal sent to the network is five times stronger when it fails to identify a pneumonia scan. This forces the optimization process to prioritize the minority class, ensuring the model's "knowledge" is biased toward safety and detection.

## 5. Backpropagation
Backpropagation is the process by which the network learns from the error calculated by the loss function. It works by traveling backward from the output layer to the first convolutional layer, using the chain rule to determine how much each individual weight and filter contributed to the final error. In a CNN, this is more complex than a standard dense network because the error must pass through the pooling and activation layers before reaching the convolutional weights. By calculating these gradients, the network determines the precise adjustment required for each parameter before the next training iteration begins.

### 5.1 The Chain Rule and Gradient Flow
To calculate how a specific weight $(w)$ should be adjusted, the network uses the Chain Rule. This mathematical principle allows us to decompose a complex relationship into a series of smaller, local derivatives. For a weight in the dense output layer, the gradient is:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

If we treat these derivatives like fractions, the intermediate terms ($\partial \hat{y}$ and $\partial z$) effectively cancel out, leaving us with the direct relationship between the Loss and the Weight ($\frac{\partial L}{\partial w}$). In practical terms, this means:

* $x_i$​: The input value arriving at the dense layer from the previous layer (a flattened feature map value).
* $w_i$: The specific weight in the dense layer connecting input $x_i$​ to the output.
* $z$: The raw weighted sum produced by the dense layer, calculated as $z = \sum_{i} w_i \cdot x_i + b$ where each weight $w_i$ is multiplied by its corresponding input $x_i$ and summed together with a bias $b$.
* $\hat{y}$: The final predicted probability, obtained by passing $z$ through the Sigmoid activation function.
* $\frac{\partial L}{\partial \hat{y}}$ **The Error Signal:** How much the loss changes as the prediction changes (the error signal)
* $\frac{\partial \hat{y}}{\partial z}$ **The Activation Slope (Sigmoid Derivative):** How much the prediction changes as the raw sum changes.
* $\frac{\partial z}{\partial w}$ **The Input Contribution:** How much the raw sum changes as the weight changes (the input contribution).

### 5.2 Backpropagating through Pooling and ReLU
When the error signal reaches a Max Pooling layer, it encounters a unique challenge: pooling layers do not have weights. Their only job during the forward pass was to select the maximum value from each $2 \times 2$ window. Because there are no weights to adjust, the layer's role in backpropagation is simply to route the error signal back to the correct location. During the forward pass, the network records an argmax mask (a temporary record of which position in each $2 \times 2$ window produced the maximum value). When the error signal travels backward, it is routed exclusively to that position. The three other pixels in each window were discarded during the forward pass and therefore had no influence on the final prediction, so they receive a gradient of zero. The gradient of the loss with respect to the input of the pooling layer $A$ is defined by the following piecewise function:

$$
\frac{\partial L}{\partial A_{i,j}} = \begin{cases} \frac{\partial L}{\partial O_{i,j}} & \text{if } A_{i,j} = \max(\text{window}) \\
0 & \text{otherwise} \end{cases}
$$

Where:  
* $\frac{\partial L}{\partial O_{i,j}}$: The incoming error signal from the next layer (the gradient with respect to the output of the pooling layer).
* $A_{i,j}$: The individual pixel in the input feature map arriving at the pooling layer.
* $\text{max}(window)$: The specific pixel that was selected as the maximum during the forward pass.

Example Argmax mask:

$$
\text{Input} = \begin{bmatrix} 
9 & 3 & 1 & 8 \\ 
2 & 6 & 5 & 3 \\ 
8 & 4 & 2 & 6 \\ 
1 & 7 & 9 & 4 
\end{bmatrix} \quad
\text{Argmax mask} = \begin{bmatrix} 
1 & 0 & 0 & 1 \\ 
0 & 0 & 0 & 0 \\ 
1 & 0 & 0 & 0 \\ 
0 & 0 & 1 & 0 
\end{bmatrix}
$$

Once the gradient has been routed through the pooling layer, it must pass through the ReLU activation before reaching the convolutional layer. ReLU backpropagation follows a straightforward rule: the gradient is passed through unchanged at positions where the forward pass activation was positive, and set to zero at positions where it was negative. This is expressed as:

$$
\frac{\partial L}{\partial z_{i,j}} = \begin{cases} \frac{\partial L}{\partial A_{i,j}} & \text{if } z_{i,j} > 0 \\
0 & \text{otherwise} \end{cases}
$$

Where:

* $\frac{\partial L}{\partial A_{i,j}}$​: The incoming error signal from the pooling layer, as computed above.
* $z_{i,j}$​: The raw pre-activation value at position $(i,j)$, computed during the forward pass.

It is this value, $\frac{\partial L}{\partial z_{i,j}}$, that is passed back into section 5.3 as the incoming error signal for computing the convolutional gradients.

### 5.3 Convolutional Gradients (Updating the Kernels)
The most critical part of the learning process is updating the convolutional filters ($K$). Unlike a standard dense layer where a weight is only responsible for one connection, a convolutional weight is reused across the entire image. Therefore, its gradient must be the sum of its contribution at every position it visited during the forward pass. To find the gradient for a specific weight within a $3 \times 3$ kernel, we use a double summation to accumulate error across the entire height and width of the feature map. Mathematically, the gradient for a kernel weight at position $(m, n)$ is:

$$
\frac{\partial L}{\partial K_{m,n}} = \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \frac{\partial L}{\partial Z_{i,j}} \cdot I_{i+m, j+n}
$$

Where:
* $\frac{\partial L}{\partial K_{m,n}}$: The total gradient for one specific weight in the $3\times3$ kernel, accumulated across every position in the feature map.
* $\sum_{i=0}^{H-1} \sum_{j=0}^{W-1}$: The spatial summations, which iterate over every row $(i)$ and column $(j)$ of the output feature map.
* $\frac{\partial L}{\partial Z_{i,j}}$: The feature map error at a specific coordinate, computed in section 5.2. It represents how much the network's output at that exact position contributed to the overall loss.
* $I_{i+m, j+n}$: The input pixel that the kernel weight was reading during the forward pass. By using the indices $i+m$ and $j+n$, we align the error with the exact patch of the image that created it.

This formula takes two things that are already known at this stage of backpropagation: the error at every position in the feature map $\frac{\partial L}{\partial z_{i,j}}$​, computed in section 5.2, and the original input image pixel values $I_{i+m,j+n}$​. By multiplying them together at every position and summing the result, we obtain the total gradient for each kernel weight. Critically, $\frac{\partial L}{\partial z_{i,j}}$ is spent twice at this layer: once here to update the kernel weights, and once in section 5.4 to pass the error signal further backwards. This is true at every convolutional layer in the network.

After the double summation is completed for all 9 weights in the kernel, we obtain a $3 \times 3$ gradient matrix, $\frac{\partial L}{\partial K}$, where each cell corresponds to the gradient of one specific weight in the kernel. This is shorthand for 9 simultaneous independent updates, one per kernel weight. For each position $(m,n)$, the update is:

$$
K_{m,n}^{\text{new}} = K_{m,n}^{\text{old}} - \eta \cdot \frac{\partial L}{\partial K_{m,n}}
$$

Written compactly for all 9 weights simultaneously:

$$
K_{\text{new}} = K_{\text{old}} - \eta \cdot \frac{\partial L}{\partial K}
$$

Each weight is adjusted independently by its own gradient. A weight that contributed heavily to the loss receives a large update, while a weight that had little influence receives a small one.

### 5.4 Passing Error to Previous Feature Maps
Once the kernel weights have been updated, $\frac{\partial L}{\partial z_{i,j}}$​ is spent for the second time now to compute how much error each input pixel in the current layer's input feature map contributed to the loss. This is necessary so that any earlier convolutional layers have an error signal to backpropagate through in turn.

To understand why this requires a flipped kernel, consider what happened during the forward pass. As the kernel slid across the input, each input pixel was touched by a different kernel weight depending on the kernel's position. Taking pixel $e$ as an example:

$$\begin{aligned}
\text{Kernel at } (0,0): & \quad e \cdot K_{1,1} \rightarrow z_{0,0} \\
\text{Kernel at } (0,1): & \quad e \cdot K_{1,0} \rightarrow z_{0,1} \\
\text{Kernel at } (1,0): & \quad e \cdot K_{0,1} \rightarrow z_{1,0} \\
\text{Kernel at } (1,1): & \quad e \cdot K_{0,0} \rightarrow z_{1,1}
\end{aligned}$$

<div align="center">
    <img src="images/kernel_sliding_over_e.jpeg" width="700">
    <p align="center"><strong>Fig. 5.</strong> As the 2×2 kernel slides to each of the four valid positions over the 3×3 input, pixel e (centre) is covered by a different kernel weight at each position. During backpropagation, the error at each output must therefore flow back to pixel e through the exact kernel weight that originally produced it. <p>
</div>

The total error flowing back to pixel $e$ is therefore calculated by summing the gradients from each output $z_{i,j}$ that $e$ contributed to, weighted by their respective kernel weights:


$$\frac{\partial L}{\partial e} = \frac{\partial L}{\partial z_{0,0}} K_{1,1} + \frac{\partial L}{\partial z_{0,1}} K_{1,0} + \frac{\partial L}{\partial z_{1,0}} K_{0,1} + \frac{\partial L}{\partial z_{1,1}} K_{0,0}$$

Notice that the kernel weights in this sum appear in reverse order: $K_{1,1}, K_{1,0}, K_{0,1}, K_{0,0}$, which is the original kernel read backwards. This reverse ordering arises directly from the forward pass: as the kernel slid from position $(0,0)$ to $(1,1)$ over pixel $e$, the weight directly above $e$ progressed from $K_{1,1}$ down to $K_{0,0}$, the opposite direction to the kernel's travel. This has an important consequence for backpropagation. If we naively convolved the error map with the original kernel, each output error would be paired with the wrong weight:

$$
\frac{\partial L}{\partial e}_{\text{wrong}} = \frac{\partial L}{\partial z_{0,0}} \times K_{0,0} + \frac{\partial L}{\partial z_{0,1}} \times K_{0,1} + \frac{\partial L}{\partial z_{1,0}} \times K_{1,0} + \frac{\partial L}{\partial z_{1,1}} \times K_{1,1}
$$

Flipping the kernel 180°, which is equivalent to reversing it both horizontally and vertically, corrects this misalignment automatically, producing the correct weight-error pairings for every input pixel simultaneously without manually tracking each connection. The general formula for the gradient with respect to the input feature map is:

$$
\frac{\partial L}{\partial I_{i,j}} = \sum_{m} \sum_{n} \frac{\partial L}{\partial z_{i-m, j-n}} \cdot K_{m,n}
$$

where the shifted indices $(i-m, j-n)$ are the mathematical expression of the kernel flip. The resulting gradient map $\frac{\partial L}{\partial I}$ then becomes the incoming error signal for the previous layer, where the same two-step process (update weights, pass error backwards) repeats again, all the way back to the first convolutional layer.

<div align="center">
    <img src="images/flipped_kernel.png" width="500">
    <p align="center"><strong>Fig. 6. </strong>Flipping the kernel 180° reverses the order of the weights, which is mathematically equivalent to changing the convolution indices from +(m, n) to -(m, n).<p>
</div>

## 6. Training & Optimization
The training process is a continuous loop of predicting, measuring error, and adjusting. While sections 3 through 5 described how a single forward and backward pass works for one image, optimization is what allows the network to learn from an entire dataset of thousands of X-rays. This section explains how the gradients computed in section 5 are applied in practice to update every learnable parameter in the network.

### 6.1 The Stochastic Gradient Descent (SGD) Loop
The network learns through a process called Stochastic Gradient Descent (SGD). Instead of processing the entire dataset at once, the model processes one small batch of images at a time. For each batch, it performs a forward pass to obtain a prediction, computes the loss using the Binary Cross-Entropy function from section 4.1, and then performs a backward pass to compute the gradients for every learnable parameter as described in section 5. The weights are then immediately updated using those gradients. By repeating this process thousands of times across the entire dataset, the model progressively reduces the total loss until it reaches the lowest point.

### 6.2 The Learning Rate ($\eta$)
The learning rate $\eta$ is a multiplier applied to every gradient before it is used to update a weight. It controls how large a step the network takes in the direction of the gradient. The general update rule is:

$$
W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}
$$

This rule is applied to every learnable parameter in the network after each backward pass. Concretely, for the CNN described in this paper, this means:

* Every weight in the convolutional kernels: $K_{m,n}^{\text{new}} = K_{m,n}^{\text{old}} - \eta \cdot \frac{\partial L}{\partial K_{m,n}}$

* Every weight in the dense layer: $w_{i}^{\text{new}} = w_{i}^{\text{old}} - \eta \cdot \frac{\partial L}{\partial w_{i}}$

* Every bias term in every layer: $b^{\text{new}} = b^{\text{old}} - \eta \cdot \frac{\partial L}{\partial b}$

If the learning rate is too high, the network may overshoot the optimal weights and fail to converge. If it is too low, the network will require a large number of epochs to reach an acceptable level of accuracy. Selecting an appropriate learning rate is therefore critical to ensuring the model converges efficiently.

### 6.3 Epochs and Convergence
Training is measured in epochs, where one epoch represents the network processing every image in the training dataset exactly once. Because the network only adjusts its weights by a small amount per batch, multiple epochs are required for the network to fully learn the features of a medical scan. As training progresses, the total loss should steadily decrease while the validation accuracy increases. When the loss plateaus and stops decreasing, the model has reached convergence.

## References
[1] Dharmaraj, "Convolutional Neural Networks (CNN) — Architecture Explained," Medium, [Online]. Available: https://owl.purdue.edu/owl/general_writing/grammar/using_articles.html.

[2] J. Starmer, "Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)," YouTube, Jan. 14, 2020. [Online]. Available: https://www.youtube.com/watch?v=HGwBXDKFk9I.