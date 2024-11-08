# 1.0 Introduction 

<p align="justify"> This case study evaluates two methods for detecting staff presence in a video: a Convolutional Neural Network (CNN) model and K-Means clustering. The purpose of the analysis is to determine which method more accurately identifies frames containing staff members, based on labeled frame data from a sample video.

![image](https://github.com/user-attachments/assets/86ceb4f5-00e3-408d-868a-757fbb5f5c18)

# 2.0 Methods
## 2.1 Convolutional Neural Network (CNN) 
<p align="justify"> The CNN model was trained to detect the presence of staff members in video frames. The model architecture included several convolutional layers followed by max-pooling, flattening, dense layers, and a dropout layer to reduce overfitting. The final output layer uses a sigmoid activation function for binary classification (staff present or not).

Model Training Details:
•	Input Size: 224x224 pixels
•	Optimizer: Adam with a learning rate of 0.001
•	Loss Function: Binary Crossentropy
•	Metrics: Accuracy
•	Training Duration: 10 epochs
•	Data Split: 80% for training, 20% for validation
<p align="justify"> The dataset was created by extracting frames from a video, labeling them as "staff_tag" or "no_staff_tag," and splitting them into training and validation sets. Data augmentation was applied to the training images to improve generalization.

## 2.2 K-Means Clustering 
<p align="justify"> K-Means clustering was used as an unsupervised learning method to detect frames with staff members. Frames were resized to 128x128 pixels and converted to grayscale before applying the K-Means algorithm to cluster the frames into two groups: one with staff and one without.


Clustering Details:
•	Number of Clusters: 2
•	Frame Representation: Each frame was reshaped into a one-dimensional vector for clustering.

# 3.0 Results
## 3.1 CNN Model Results
•	Training Accuracy (last epoch): 87.79%
•	Validation Accuracy (last epoch): 81.25%
•	Number of Frames with Staff Detected by CNN: 1341
<p align="justify"> The CNN model detected staff presence in 1341 frames. The training accuracy was relatively high, but the validation accuracy was significantly lower, indicating potential overfitting or a lack of generalization to unseen data. This discrepancy suggests the model might have learned to classify the training set well but struggled with new, unseen frames.

## 3.2 K-Means Clustering Results
•	Number of Frames with Staff Detected by K-Means: 874
<p align="justify"> K-Means clustering identified 874 frames with staff presence. As an unsupervised learning method, K-Means does not rely on labeled data and instead divides the data into two groups based on inherent similarities.

# 4.0 Analysis and Discussion
•	Accuracy and Generalization: The CNN model achieved high accuracy during training but exhibited lower validation accuracy, indicating overfitting. This means that while the CNN was effective at learning the training data, it struggled with generalizing to new frames. In contrast, K-Means clustering does not have a notion of accuracy in the same sense but can provide an initial grouping that can be useful for exploratory analysis.
•	Number of Frames Detected: The CNN model detected more frames (960) with staff presence than K-Means (874). However, due to the low validation accuracy, it is possible that some frames were falsely classified by the CNN as containing staff.
•	Utility of Each Method: CNNs are powerful tools for image classification when labeled data is available and the model is sufficiently trained. K-Means, on the other hand, is a useful alternative when labeled data is scarce or when exploring the structure of the data. In this case, the K-Means results might help identify patterns that the CNN model struggled with.

# 5.0 Conclusion
<p align="justify"> The CNN model and K-Means clustering each have strengths and limitations in detecting staff presence in video frames. The CNN model achieved high training accuracy but struggled with generalization, as evidenced by the low validation accuracy. K-Means clustering provided a viable alternative for exploring the data without the need for labels but lacks the predictive power of a well-trained CNN. Future improvements could include further tuning the CNN model to avoid overfitting, perhaps by employing techniques such as cross-validation, adding more diverse training data, or fine-tuning the model architecture.



