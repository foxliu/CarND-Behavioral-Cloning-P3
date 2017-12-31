
# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nVidea_data.png "NVIDIA Architecture"
[image2]: ./examples/origin_hist.png "origin hist"
[image3]: ./examples/result_hist.png "result hist"
[image4]: ./examples/origin_image.png "origin image"
[image5]: ./examples/normaled_image.png "normaled Image"
[image6]: ./examples/crop_image.png "crop Image"
[image7]: ./examples/train_plot.png "train loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode in the simulator(This is provided Udacity, my only modification was to increatse the car speed on line 47 from 9 to 25
* model.h5 containing a trained convolution neural network 
* writeup_report.md : summarizing the results

Node:

On my first iteration, I tried LeNet model and nVidia Autonomous Car Group model. This experiments could be found in class Model.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My initial approach was to use LeNet, but it was hard to have the car inside the street with three epochs (this model could be found here). After this, I decided to try the nVidia Autonomous Car Group model, and the car drove the complete first track after just three training epochs (this model could be found here).

A model summary is as follows:
```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 80, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 6336)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           633700      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
```

#### 2. Attempts to reduce overfitting in the model

I decided not to modify the model by applying regularization techniques like Dropout or Max pooling. Instead, I decided to keep the training epochs low: only three epochs. In addition to that, I split my sample data into training and validation data. Using 70% as training and 30% as validation. 

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Also, the data provided by Udacity, I used the first track and second track data. The simulator provides three different images: center, left and right cameras. Each image was used to train the model.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to try the LeNet model with three epochs and the training data provided by Udacity. On the first track, the car went straight to the lake. I needed to do some pre-processing. A new Lambda layer was introduced to normalize the input images to zero means. This step allows the car to move a bit further, but it didn't get to the first turn. Another Cropping layer was introduced, and the first turn was almost there, but not quite.

The second step was to use a more powerfull model: nVidia Autonomous Car Group The only modification was to add a new layer at the end to have a single output as it was required. This time the car did its first complete track, but there was a place in the track where it passes over the "dashed" line. More data was needed. Augmented the data by adding the same image flipped with a negative angle. In addition to that, the left and right camera images where introduced with a correction factor on the angle to help the car go back to the lane. After this process, the car continues to have the same problem with the same "dashed" line. I needed more data, but it was a good beginning.


The following is the processing of the image：
First Steering angles histogram
![alt text][image2]

sample the data set use the code in model.py line 42-55
the result
![alt text][image3]

Preprocess image:
The data image:
![alt text][image4]

Normaled image use lambda x: x/255.0 - 0.5:
![alt text][image5]

Croped image use Cropping2D(cropping=((60, 20), (0, 0))):
![alt text][image6]


#### 2. Final Model Architecture

The final model architecture is shown in the following image:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

All these data was used for training the model with three epochs. The data was shuffled randomly. The following picture shows the training:

![alt text][image7]
