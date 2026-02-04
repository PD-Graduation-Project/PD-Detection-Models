# SpiralDrawings Dataset - Model Selection

The SpiralDrawings dataset contains **3264 images**, which is a moderate dataset size.  
Given this size, we opted for **moderate-sized models** with transfer learning and heavy augmentation to achieve good performance while avoiding overfitting.

## Models Chosen

- **DenseNet161** – 28.7M parameters  
  Moderate depth to extract meaningful features without being too heavy.

- **EfficientNet-B2** – 9.2M parameters
  Slightly larger than B1, offering more capacity while still efficient for moderate dataset sizes.


- **InceptionV3** – 23.8M parameters  
  Larger model but used effectively in literature; provides diverse feature extraction.

- **MobileNetV3-Large** – 5.4M parameters  
  Efficient and fast, captures complementary features.

- **ResNet34** – 21.8M parameters  
  Balanced depth and capacity, suitable for moderate-sized datasets.
