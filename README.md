# Melanoma-Skin-Cancer-Detection

## **Abstract**
In the realm of cancer, there exist over 200 distinct forms, with melanoma standing out as the most lethal type of skin cancer among them. The diagnostic protocol for melanoma typically initiates with clinical screening, followed by dermoscopic analysis and histopathological examination. Early detection of melanoma skin cancer is pivotal, as it significantly enhances the chances of successful treatment. The initial step in diagnosing melanoma skin cancer involves visually inspecting the affected area of the skin. Dermatologists capture dermatoscopic images of the skin lesions using high-speed cameras, which yield diagnostic accuracies ranging from 65% to 80% for melanoma without supplementary technical assistance. Through further visual assessment by oncologists and dermatoscopic image analysis, the overall predictive accuracy of melanoma diagnosis can be elevated to 75% to 84%. The objective of the project is to construct an automated classification system leveraging image processing techniques to classify skin cancer based on images of skin lesions.

## **Problem statement**
The goal is to develop a CNN-based model capable of accurately detecting melanoma, a type of cancer that can be fatal if not identified early, responsible for 75% of skin cancer-related deaths. A system that can analyze images and notify dermatologists about the presence of melanoma has the potential to significantly reduce the manual effort involved in diagnosis.

## **Table of Contents**
1. General Information
2. CNN Architecture Design
3. Model Summary
4. Model Evaluation
5. Technologies Used


## **General Information**
The dataset consists of 2,357 images representing both malignant and benign oncological conditions, obtained from the International Skin Imaging Collaboration (ISIC). These images were organized according to the classification provided by ISIC, with each category containing an equal number of images.

<img width="762" alt="Screenshot 2025-03-19 at 4 37 32 PM" src="https://github.com/user-attachments/assets/8a6f21e4-16ca-4686-bb73-e876f1580f04" />


In order to address the challenge of class imbalance, the Augmentor Python package (https://augmentor.readthedocs.io/en/master/) was employed to augment the dataset. This involved generating additional samples for all classes, ensuring that none of the classes had insufficient representation.

### **Pictorial representation of skin types**
<img width="683" alt="Screenshot 2025-03-19 at 4 36 44 PM" src="https://github.com/user-attachments/assets/df2cf606-e509-472d-9c43-cfec577b4859" />

The aim of this task is to assign a specific class label to a particular type of skin cancer.

## **CNN Architecture Design**
Here is a step-by-step breakdown of the final CNN architecture:

1. Convolutional Layers: Three convolutional layers are added sequentially using the Conv2D function. Each convolutional layer is followed by a ReLU activation function, introducing non-linearity to the model. The padding='same' argument ensures that the spatial dimensions of the feature maps remain unchanged after convolution. The numbers in each Conv2D layer (16, 32, 64) correspond to the number of filters or kernels, which determine the depth of the feature maps.

2. Data Augmentation: The augmentation_data variable represents the techniques applied to enhance the training data. Data augmentation artificially expands the dataset by applying random transformations like rotation, scaling, and flipping to the images. This process helps improve the model's generalization ability.

3. Dropout Layer: A Dropout layer with a rate of 0.2 is added after the final max-pooling layer. Dropout is a regularization technique that prevents overfitting by randomly deactivating a fraction of the neurons during training.

4. Flatten Layer: The Flatten layer is included to convert the 2D feature maps into a 1D vector, preparing the data for input into the fully connected layers.

5. Fully Connected Layers: Two fully connected (dense) layers are added, each with ReLU activation functions. The first dense layer has 128 neurons, and the second dense layer generates the final classification probabilities for each class.

6. Model Compilation: The model is compiled using the Adam optimizer (optimizer='adam') and the Sparse Categorical Crossentropy loss function (loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)), which is ideal for multi-class classification problems. Accuracy is selected as the evaluation metric (metrics=['accuracy']).

7. Normalization: A Rescaling(1./255) layer is added to normalize the pixel values of the input images. This normalization scales the pixel values to a range between 0 and 1, which stabilizes the training process and accelerates convergence.

8. Output Layer: The number of neurons in the output layer is determined by the target_labels variable, which indicates the number of classes in the classification task. This layer doesn't have an activation function, as it is followed by the loss function during training.

9. Pooling Layers: After each convolutional layer, a max-pooling layer (MaxPooling2D) is applied to downsample the feature maps, reducing their spatial dimensions while retaining important information. Max-pooling helps reduce computational complexity and mitigate overfitting.

10. Training: The model is trained using the fit method for a specified number of epochs (epochs=50). The ModelCheckpoint and EarlyStopping callbacks are used to monitor validation accuracy during training. The ModelCheckpoint callback saves the model with the best validation accuracy, while the EarlyStopping callback halts training if validation accuracy doesn't improve after a set number of epochs (patience=5). These callbacks help prevent overfitting and ensure the model converges to the best solution.


## **Model Summary**
<img width="983" alt="Screenshot 2025-03-19 at 4 34 58 PM" src="https://github.com/user-attachments/assets/78eeaae6-2b8a-4bc9-8708-4f1514dbf176" />


## **Model Evaluation**
<img width="953" alt="Screenshot 2025-03-19 at 4 35 25 PM" src="https://github.com/user-attachments/assets/f2d4682d-13da-4f07-bab6-bf8554fcddd3" />


## **Technologies Used**
- Python - version 3.11.4
- Matplotlib - version 3.7.1
- Numpy - version 1.24.3
- Pandas - version 1.5.3
- Seaborn - version 0.12.2
- Tensorflow - version 2.15.0

