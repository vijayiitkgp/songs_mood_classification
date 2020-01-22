# Songs Mood Classification

Find the mood of the audio songs based on the given audio features. In this project only two moods have been taken care of, HAPPY and SAD.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for training the mmodel and for predicting purposes.

### Prerequisites

Your system should have python 3.0+ installed.


### Installing

The dependencies are mentioned in the requirement.txt file.

Step to install dependencies:

pip3 install -r requirement.txt

## Train the model

Activate your virtual environment, having above above mentioned dependencies installed.

Go to commond line and go to the root directory of the project. Run the following command:

pyhton3 training.py -f <training_file_name>.csv

Note: Give the file in .csv format only. <training_file_name>.csv file will contain the data to train the model.

Above command will return the path where trained model has been saved.

Note: Training will take the time as ensemble techniques has been used for classification purpose.

### Get Prediction using saved model

Activate your virtual environment, having above above mentioned dependencies installed.

Go to commond line and go to the root directory of the project. Run the following command:

pyhton3 prediction.py -f <evaluation_file_name>.csv -m <saved_model_file_path>

Note: Give the file in .csv format only. <evaluation_file_name>.csv file will contain the data to evaluate the trained model.

Above command will override the given file and will add "MOOD_TAG" column in the file which will hold the prediction values.

