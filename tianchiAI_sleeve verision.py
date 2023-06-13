import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import os
import pandas as pd

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Change the directory
os.chdir("/Users/wendyma/Documents/MSBA/practicum/image classification/train/Images")

# Get labels
labels1 = pd.read_csv("/Users/wendyma/Documents/MSBA/practicum/image classification/round1_train/Annotations/label.csv", names=["path", "folder", "code"])
labels2 = pd.read_csv("/Users/wendyma/Documents/MSBA/practicum/image classification/round2_train/Annotations/label.csv", names=["path", "folder", "code"])
labels = pd.concat([labels1, labels2], ignore_index=True)
labels.head()

'''
# Extract the directory and the image name
for i in range(len(labels)):
    split = os.path.split(labels.loc[i,'path'])
    labels.loc[i,'directory'] = split[0]
    labels.loc[i,'filename'] = split[1]
labels.head()
'''

# Expand the path to absolute path
path0 = "/Users/wendyma/Documents/MSBA/practicum/image classification/train/"
for i in range(len(labels)):
    labels.loc[i,'path'] = path0 + labels.loc[i,'path']
labels.head()

# Subset annotations
# lapel_data = labels[labels['folder'] == 'lapel_design_labels'].reset_index(drop=True)
# neck_data = labels[labels['folder'] == 'neck_design_labels'].reset_index(drop=True)
sleeve_data = labels[labels['folder'] == 'sleeve_length_labels'].reset_index(drop=True)

# Create lists for each hierarchy - lapel design, neck design
# lapel = ["Invisible", "Notched", "Collarless", "Shawl Collar", "Plus Size Shawl"]
# neck = ["Invisible", "Turtle Neck", "Ruffle Semi-High Collar", "Low Turtle Neck", "Draped Collar"]
sleeve = ["Invisible","Sleeveless","Cup Sleeves","Short Sleeves","Elbow Sleeves","3/4 Sleeves","Wrist Length","Long Sleeves","Extra Long Sleeves"]


# Check if no-y exists
for i in range(len(labels)):
    if "y" not in labels.loc[i,'code']:
        print("yes", i)
        break
#  NO SUCH SITUATION - NO OUTPUT IN THE TERMINAL



## Decode label 
def decode(data, list):
    for i in range(len(data)):
        pos = data.loc[i,'code'].find('y')
        data.loc[i,'label'] = list[pos]


## Split the data
def Gen(data):
    datagen = ImageDataGenerator(rescale=1./255.,validation_split = 0.2)
    IMG_DIM = 224
    BATCH_SIZE = 32

    train_ds = datagen.flow_from_dataframe(dataframe=data,
                                                target_size=(IMG_DIM,IMG_DIM),
                                                x_col='path',
                                                y_col='label',
                                                class_mode="categorical",
                                                batch_size=BATCH_SIZE,
                                                subset="training",
                                                #validate_filenames=False,
                                                seed=117,
                                                shuffle=True)

    val_ds = datagen.flow_from_dataframe(dataframe=data,
                                                target_size=(IMG_DIM,IMG_DIM),
                                                x_col='path',
                                                y_col='label',
                                                class_mode="categorical",
                                                batch_size=BATCH_SIZE,
                                                subset="validation",
                                                #validate_filenames=False,
                                                seed=117,
                                                shuffle=True)

    return train_ds, val_ds

## Run the model
def RUN(train_ds, val_ds, category, epoch = 30, earlystop = 10):
    ## Build the modle
    n_classes = len(train_ds.class_indices)

    model = Sequential([
        #hub.KerasLayer("C://UCD/__Practicum/ImageTagging/imagenet_resnet_v2_50_classification_5",
        hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5",
        # hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5",
        # hub.KerasLayer("C://UCD/__Practicum/ImageTagging/imagenet_inception_resnet_v2_classification_5",
        trainable = True,
        arguments = dict(batch_norm_momentum = 0.997)),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(n_classes, activation='softmax')
        ])
    model.build([None, 224, 224, 3])
    model.summary()

    ## Complie the model
    model.compile(optimizer='adam',
              loss="categorical_crossentropy",     # SparseCategoricalCrossentropy used for interger Yi; CategoricalCrossentropy used for one-hot Yi
              metrics=['accuracy'])
    
    ## Fit the model
    epochs = epoch

    checkpoint = ModelCheckpoint(f"weights_{category}.hdf5", monitor='val_accuracy', mode="max", verbose = 1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=earlystop)
    callback_list = [checkpoint, early_stopping]

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callback_list
    )

    return history

## Visualization
def Vis(history, category):
    ## Visualize the output
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{category}: Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{category}: Training and Validation Loss')
    plt.savefig(f'{category}.png')
    plt.show()



# # Lapel_Design
# lapel_train, lapel_val = Split("lapel_design_labels")
# #lapel_train = dataAug(lapel_train)
# hist_lapel = RUN(lapel_train, lapel_val, "Lapel_Design_ResNetV2", False, False, 10)
# Vis(hist_lapel, "Lapel_Design")
# hist_lapel_tr = RUN(lapel_train, lapel_val, "Lapel_Design_ResNetV2_Tr", False, True, 20)
# Vis(hist_lapel_tr, 'Lapel_Design_tr')
# hist_lapel_incp = RUN(lapel_train, lapel_val, "Lapel_Design_IncpResNetV2", True, False, 10)
# Vis(hist_lapel_incp, 'Lapel_Design_incp')
# hist_lapel_incp_tr = RUN(lapel_train, lapel_val, "Lapel_Design_ResNetV2_Tr", True, True, 20)
# Vis(hist_lapel_incp_tr, 'Lapel_Design_incp_tr')

# # Neck_Design
# neck_train, neck_val = Split("neck_design_labels")
# #neck_train = dataAug(neck_train)
# hist_sleevehist_neck = RUN(neck_train, neck_val, "Neck_Design_ResNetV2", False, False, 10)
# Vis(hist_neck, "Neck_Design1") 
# hist_neck_tr = RUN(neck_train, neck_val, "Neck_Design_ResNetV2_Tr", False, True, 20)
# Vis(hist_neck_tr, 'Neck_Design_tr')


# sleeve_data.head()
# sleeve_data['label'].unique()
# sleeve_data.shape
# print(sleeve_data['label'].unique())
decode(sleeve_data, sleeve)
sleeve_data
sleeve_data = sleeve_data[sleeve_data['label'] != 'Invisible']
sleeve_data = sleeve_data.replace(["Extra Long Sleeves", "Wrist Length","3/4 Sleeves","Cup Sleeves"], ["Long Sleeves","Long Sleeves","Elbow Sleeves","Short Sleeves"])
# sleeve_train, sleeve_val = Gen("sleeve_length_labels")
# #sleeve_train = dataAug(neck_train)
# #hist_neck = RUN(neck_train, neck_val, "Neck_Design_ResNetV2", False, False, 10)
 # #Vis(hist_neck, "Neck_Design1")
# Vis(hist_sleeve_tr, 'Sleeve_Length_tr')
sleeve_train, sleeve_val = Gen(sleeve_data)
hist_sleeve = RUN(sleeve_train, sleeve_val, "sleeve_Design")
Vis(hist_sleeve, "sleeve_Design")
