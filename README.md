# image_classification

This is a project for training a deep learning model for accomplishing an clothing attribute classification job.
The dataset for training and testing the model is from FashionAI Attributes Recognition of Apparel Dataset. The dataset can be downloaded directly from the link as long as you’ve registered for a free account.

There are two training datasets designed for round 1 and round 2 of the competition. Since the categories and labels are the same, I combined the images in these two datasets together as a new dataset. The training job is done baded on the integrated dataset.

# Updates:
printClassifier.ipynb contains the comprehensive process of performing image classification for classifying prints of clothes from splitting the data to evaluate the model and future steps using Python TensorFlow.

AttributeDescription.md displays the information about the attributes contained in the dataset. There are 8 categories in the dataset, and 5 - 10 attributes for each category. Below are the categories and corresponding labels that we look at:

Coat Length: Invisible, High Waist Length,Regular Length,Long Length,Micro Length, Knee Length, Midi Length, Anckle&Floor Length

Collar Design: Invisible, Shirt Collar, Peter Pan, Puritan Collar, Rib Collar

Lapel Design: Invisible, Notched, Collarless, Shawl Collar, Plus Size Shawl

Neck Design: Invisible, Turtle Neck, Ruffle Semi-High Collar, Low Turtle Neck, Draped Collar

Neckline Design: Invisible, Strapless Neck, Deep V Neckline, Straight Neck, V Neckline, Square Neckline, Off Shoulder, Round Neckline, Swear Heart Neck, One Shoulder Neckline

Pant Length: Invisible, Short Pant, Mid Length, 3/4 Length, Cropped Pant, Full Length

Skirt Length: Invisible, Short Length, Knee Length, Midi Length, Ankle Length, Floor Length

Sleeve Length: Invisible, Sleeveless, Cup Sleeves, Short Sleeves, Elbow Sleeves, 3/4 Sleeves, Wrist Length, Long Sleeves, Extra Long Sleeves

main.py is the main part of the image classification training job. This python script file mainly performs jobs mentioned below:

Load data and Split the data into training and validation set;

Five Data Augmentation layers for choice including RandomFilp, RandomRotation, RandomTranslation, RandomBrightness, and RandomContrast;

Use transfer learning to train the model. The default model selection for transfer learning is the ResNet v2 model of 50 layers without fine-tuning, while Inception ResNet v2 model and Fine-tune are optional. The model is trained separately for different categories;

Store the weights of the model which has the best performance on the validation set for future use;

Visualize the training process by ploting the changes in training, validation loss and accuracy over epoches.

Preprocess.py is used for preprocessing the dataset by grouping images based on labels into subfolders and combining labels based on the distribution and nature of the labels. Main steps to accomplish the task are listed below:

As mentioned above, combine round 1 and round 2 training set together as an integrated training set;

Decode the original labels to eligible ones: Assign corresponding labels based on the position of "y" in the encoded labels. For example, "nnynn" for "Collar Design" category denotes the label "Peter Pan" since its the third label in the category;

Create subfolders for each category based on the labels and group images to corresponding folders. This step is designed for using "image_dataset_from_directory" rather than "ImageDataGenerator" for a faster and more efficient training process;

Functions for checking the distribution of labels of certian category and preview images of certain label. This step is designed for examining the distribution and nature of labels to help make decisions on what labels to combine;

The "regroup" function implement the decisions made for categories by rearrange the subfolders each image belong to. The reason why I combine labels is because the boundaries among labels provided are not very clear in all cases. Even for human being, some labels are difficult to determine using the tagging system with this level of granularity. So this level of granularity is not desired for this project. Also, with fewer output labels, the accuracy of the model would increase inherently.
