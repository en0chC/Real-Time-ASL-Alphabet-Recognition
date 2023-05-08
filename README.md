# Real-Time-ASL-Alphabet-Recognition

QuAM_app.py - Run this file to run the QuAM interface. This will activate and
show your webcam videostream with a green rectangle in the middle. Perform ASL
alphabet handsigns within this rectangle and the videostream will display its
predicted class label. Press key 'q' to stop the videostream. Note that the 
classifier doesn't work well with complex backgrounds. For higher accuracy,
use a single color background, preferably white.

QuAM_report.ipynb - Jupyter notebook containing the D3 report.

CNN_model file - Contains the saved CNN model that is loaded and used by 
QuAM_app.py.

labels_clean.csv - CSV file containing the class labels of all the images in the 
images_dataset_grayscale file with no NaNs.

labels.csv - CSV file containing the class labels with NaNs.

images_dataset_grayscale file - Contains all the images used by QuAM_report.ipynb
to train the models.

images_dataset_test_1 file - Contains images used by QuAM_report.ipynb to test the
accuracy of the models.

images_dataset_test_2 file - Contains more images used by QuAM_report.ipynb to 
test the accuracy of the models.
