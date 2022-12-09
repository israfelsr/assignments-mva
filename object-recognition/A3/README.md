# Assignment 3: Fine-Grained Image Recognition

## Goal
This assignment requires you to participate in a Kaggle competition with the rest of the class on the Caltech-UCSD Birds-200-2011 bird dataset. The objective it to produce a model that gives the highest possible accuracy on a test dataset containing the same categories.

## Guidelines
You should first clone the Github repository https://github.com/willowsierra/recvis22_a3.
The main.py file contains code for training and evaluating your models. Once training is completed, it produces a file kaggle.csv that lists the IDs of the test set images, along with their predicted label. This file should be uploaded to the Kaggle webpage, which will then produce a test accuracy score. You can register for the

## What to hand-in
You should write a 1-page, double-column report in CVPR’23 format briefly presenting your approach and obtained results. This report should be uploaded in pdf format to the course Google Classroom by November 29th 2022.
The model architecture is specified in model.py. Currently a simple baseline model is provided. You can use this model or create your own. You are free to implement any approach covered in class, or in the research literature. Of course, tricks that you devise yourself are also encouraged. The test and training data are provided in the Kaggle
competition data download page. You should unzip the images.

## External dataset policy
1. Solutions should not use external datasets with annotation for 10+ species of birds such
as NABirds or iNaturalist.
2. Solutions should not use pre-trained models learned on datasets in 1.
3. Solutions can use any datasets or self-collected images without annotation, including
images from NABirds or iNaturalist -without- labels.
4. Solutions can use ImageNet pre-trained checkpoints.