# Flower classification

### **1. Objective and Goal of the Project**

#### **Objective:**
The goal of this project is to build a machine learning model to classify flower images into categories. The dataset used contains 5 different classes of flowers: Daisy, Dandelion, Roses, Sunflowers, and Tulips. The model will be trained to predict the class of a given flower image based on its visual features.

#### **Goal:**
- To train an image classification model that achieves high accuracy in predicting flower categories.
- To implement preprocessing techniques for resizing, augmenting, and splitting the dataset into training and testing sets.
- To fine-tune the model for optimal performance using techniques like dropout for regularization.
- To deploy the model and create an interactive interface that allows users to upload an image and get predictions on flower types.

---

### **2. Model Used**

#### **Model Architecture:**
The model used for this classification task is a **Convolutional Neural Network (CNN)**, which is well-suited for image classification tasks due to its ability to detect patterns and spatial hierarchies in images.

Here is a summary of the architecture:

1. **Input Layer:**
   - The images are resized to 128x128 or 180x180 pixels and normalized using `Rescaling(1./255)` to ensure the pixel values are between 0 and 1.

2. **Convolutional Layers:**
   - **Conv2D Layers:** The model uses several convolutional layers (with 32, 64, and 128 filters) with ReLU activations to extract features from the images.
   - **MaxPooling2D:** Pooling layers are used to reduce the spatial dimensions of the image while retaining important features.

3. **Fully Connected Layers:**
   - The output from the convolutional layers is flattened into a 1D vector and passed through fully connected layers.
   - The final layer is a `Dense` layer with 5 units (since there are 5 flower classes) and a `softmax` activation function to predict the probability distribution over the classes.

4. **Regularization:**
   - **Dropout:** A dropout layer is added to prevent overfitting by randomly setting a fraction of the input units to 0 during training.

5. **Loss Function and Optimizer:**
   - The model uses **categorical crossentropy** as the loss function (because it's a multi-class classification problem) and the **Adam optimizer** to minimize the loss.

--

### **3. Working Process**

#### **Data Preprocessing:**
1. **Dataset Loading:** 
   - The flower images are downloaded from an online source and extracted into a directory structure where each class has its own folder containing images of flowers.
  
2. **Resizing and Normalization:** 
   - All images are resized to a standard size (128x128 or 180x180) to ensure uniformity.
   - The images are then normalized to have pixel values between 0 and 1 by dividing by 255.

3. **Data Splitting:**
   - The dataset is split into training and testing sets using an 80-20 split. The `train_test_split` function ensures that the images from each class are distributed evenly between the training and testing sets.

4. **Data Augmentation (Optional):**
   - To prevent overfitting and increase the diversity of the dataset, **ImageDataGenerator** can be used to perform real-time data augmentation such as rotating, zooming, and shifting images.

#### **Model Training:**
- The model is trained on the training dataset for multiple epochs (e.g., 15 epochs). The validation data is used to evaluate the performance of the model after each epoch.
- **Early stopping** is used to stop the training process when the validation loss stops improving, preventing unnecessary computation.

#### **Evaluation:**
- After training, the model is evaluated on the test set, and the accuracy is reported to assess its generalization performance.


### **4. Model Deployment**

#### **Deployment Process:**
- **Gradio Interface:**
   - A Gradio interface is used to deploy the model in an interactive web application. Users can upload an image, and the model will predict the flower class.
   - The Gradio interface provides the model's predictions for each of the 5 flower classes, showing the probabilities for each class.
   

#### **Gradio Interface Features:**
- **Image Input:** Users can upload images of flowers.
- **Label Output:** The interface outputs the predicted class of the flower along with the probabilities for each class.

### **5. Final Steps and Conclusion**

- The model is successfully trained and evaluated on the flower dataset.
- The interactive Gradio interface provides a user-friendly deployment solution where anyone can upload an image and get predictions.
- This flower classification model can be extended with more advanced techniques like transfer learning (e.g., using ResNet50) for better performance on larger datasets.

By following this process, the project achieves its goal of creating a functional, deployed flower classification model with good accuracy and user accessibility.
