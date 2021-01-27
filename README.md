# Image-Classification-of-Drawings

This code repo was build as a part of the kaggle competition 2(https://www.kaggle.com/c/ift3395-6390-quickdraw Classification of drawings) for IFT 6390 Machine Learning course. This is a image classification task in which Convolutional Neural Network model was built using tensorflow and keras to classify images of drawings.

Prerequisite:

1. File path

i.  Make sure the datasets are having the proper data file path when you are loading it in kaggle. The file path should be :- 
../input/ift3395-6390-quickdraw/train.npz and ../input/ift3395-6390-quickdraw/test.npz

ii. If the data path file is different, rename the path file if required. This could happen if you are running the notebook on colab, local machine, etc.

2. The codes require numpy, sklearn, tensorflow and keras libraries to run.

Steps to run the code:

Step 1: run the python file ".py" as python file_name.py

Step 2: A csv file will generated which is the model prediction on the test data

Step 3: Upload the csv file created on Kaggle

There are two .py files :-

1. File name- ml_cnn.py

python ml_cnn.py

Public Score = 0.85183
Private Score = 0.85457

Submission file- ml_cnn.csv

Note: This file contains 1 CNN model. It's faster and easier to run it and it is recommended to use this if the goal is to get submission file that scores higher than Random prediction and TA Baseline. 
The leaderboard score after running this file could have a difference of +/- 0.005 than the Public(0.85183) and Private Score(0.85457).

2. File name- ml_ensemble.py

python ml_ensemble.py

Public Score = 0.87222
Private Score = 0.87607

Submission file- ml_ensemble.csv

Note: This file contains an ensemble of 4 CNN models and this code produces the csv file with the best performance. It's slower to run it because it builds 4 CNN models and it is recommended to use this if the goal is to get a submission file that performs equal to our best leaderboard score. This code was selected as our final submission in the Kaggle Competition. 
The leaderboard score after running this file could have a difference of +/- 0.005 than the Public(0.87222) and Private Score(0.87607).
