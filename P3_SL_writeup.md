# **Traffic Sign Recognition** 

## Writeup
### Project 3. "Becoming a Self-Driving Car Engineer" Nanodegree (Udacity)
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Submission_images/BarChart.png "Data Bar chart Visualization"
[image2]: ./Submission_images/greyscale.png "Grayscale comparison"
[image3]: ./Submission_images/Normalise.png "Normalizing the data"
[image4]: ./Submission_images/1.png "Traffic Sign 1"
[image5]: ./Submission_images/2.png "Traffic Sign 2"
[image6]: ./Submission_images/3.png "Traffic Sign 3"
[image7]: ./Submission_images/4.png "Traffic Sign 4"
[image8]: ./Submission_images/5.png "Traffic Sign 5"
[image9]: ./Submission_images/6.png "Traffic Sign 6"
[image10]: ./Submission_images/Probability.png "Traffic Sign Probability 1"
[image11]: ./Submission_images/Probability2.png "Traffic Sign Probability 2"
[image12]: ./Submission_images/Probability3.png "Traffic Sign Probability 3"
[image13]: ./Submission_images/Probability4.png "Traffic Sign Probability 4"
[image14]: ./Submission_images/Probability5.png "Traffic Sign Probability 5"
[image15]: ./Submission_images/Probability6.png "Traffic Sign Probability 6"

## Rubric Points
### (https://review.udacity.com/#!/rubrics/481/view) Points

This writeup is the explantation response of my code to the rubric criteria (above).

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/SamaraLove/Traffic_Light_Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The German Traffic Signs dataset was visualised by using the pickle files. The statistics of the traffic signs data set was found through using python to read in the features and labels from the 3 original files. 

* The size of training set is 34799
* The size of the validation set is 4410 
* The size of test set is 12630
* Image data shape = (32, 32, 3)
* The number of unique classes/labels in the data set is 43

One random example image was printed to the screen with its corresponding label.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data across the 43 different label classes. It is noted there is an imbalance accross the classes in the training set. To avoid bias in the model towards the extra represented classes, it is recognised that fake data could have been generated, but it wasn't necessary for this project to achieve the required accuracy.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

It was originally tested without any preprocessing and using the original LeNet Lab example, and as the project suggests, an accruacy of 0.89 was consistently achieved.

The image list was then converted to grayscale using 'gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )'. The images were converted to gray scale because the color dimensions add complexity and might not provide much information. 

Example of 60 speed sign with RGB and grey image
![alt text][image2]


Then, the training data was run through again and increased the accuracy by 0.02 on average. 

The data set was then normalized to have a mean of 0 and unit variance. The normalization makes training less sensitive to the scale of features, so we can better solve for coefficients. To apply the normalisation and grayscale, I altereed the grayscale function to return the data/3 so that the lists can be passed into the normalisation function without error. 
The following was printed to the screen:
RGB image shape:  (32, 32, 3)
Gray image shape:  (32, 32, 1)
Note that the 3 has been converted to a 1

Example of the normalisation function is below:

![alt text][image3]

The data set was run again with both preprocessing features and an accuracy of 0.93 was acheived, as desired. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model, see below for the list of layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grey image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | Output=10x10x16 								|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  output 5x5x16 					|
| Flatten				| Input=5x5x16, output=400						|
| Fully connected		| input=400, output=120							|
| RELU					|												|
| Fully connected		| input=120, output=84							|
| RELU					|												|
| Fully connected		| input=84, output=43							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The learning rate and batch size remained constant. 

With each change in the preprocessing the model was trained with epochs of 10,20,25 and 30 to compare results. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.93 
* test set accuracy of 0.903

* Which parameters were tuned? How were they adjusted and why?
    The RGB to Grey was adjusted in the architecture, depending on the type of image I was feeding in. 
    The epochs were increased in the model was 
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
    LeNet was chosen as it had been proven in the lectures and tutorials and so I was most familiar with this method. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    As mentioned above, the training data nad valdiation accuracy was fairly high and thus acheived the required accuracy. The test accuracy was lower, but I did choose some images that I did not account for in the preprocessing (such as big changes in light) and thus I expected that they would not be determined correctly. 
    It is recognised that the training set and valdiation sets had some differences in their accuracies, which may be due to not enough adjsutment to counteract the overfitting or underfitting, but it was altered as the required validation accuracy was acheived. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 random German traffic signs that were found on the web:


![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

They were chosen as they have similarities and have different positions and amounts of light. I expected the stop sign with the most white light would be difficult to determine the corresponding label correctly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
|30 km/h	      		| 60 km/h    					 				|
|30 km/h	      		| 30 km/h    					 				|
| Yield					| Yield											|
| Stop 					| Yield 										|
| Keep Left				| Keep Left 									|


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 67%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the "Output Top 5 Softmax Probabilities For Each Image Found on the Web" section of my iPython notebook. 

The top 5 softmax probabilities of the 6 images are shown below. It was correctly certain for 4/6. The second stop sign had a high liklihood be being misinterpretated as a yied sign. The first 30 km/h sign was 100% determined as a 60 km/h sign instead of 30km/h. 

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14] ![alt text][image15]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


