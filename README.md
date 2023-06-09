# Real-Time-ASL-Alphabet-Recognition

QuAM_app.py - Run this file to run the QuAM interface. This will activate and
show your webcam videostream with a green rectangle in the middle. Perform ASL
alphabet handsigns within this rectangle and the videostream will display its
predicted class label. Press key 'q' to stop the videostream. Note that the 
classifier doesn't work well with complex backgrounds. For higher accuracy,
use a single color background, preferably white.

DataPreparation.ipynb - Jupyter notebook containing the D2 data preparation report.

QuAM_report.ipynb - Jupyter notebook containing the D3 model training report.

CNN_Model Google Drive Link.md  - Contains the Google Drive link of the saved CNN model 
that is loaded and used by QuAM_app.py.

labels_ohe.csv - CSV file containing the one hot encoded class labels of all the 
images in the images_dataset_grayscale file with no NaNs.

labels_clean.csv - CSV file containing the class labels of all the images in the 
images_dataset_grayscale file with no NaNs.

labels.csv - CSV file containing the class labels with NaNs.
