# DNN_for_German_traffic_sign

This is an application of deep fully connected neural network for German traffic sign recognition. 
This is a part of the overall pipeline of traffic sign recoginition. 

Steps: 
  Give region of interest of traffic signs in images(Roi.x1, Roi.x2, Roi.y1, Roi.y2) 
  1. reshape the roi into 32 by 32 images 
  2. vectorize the image matrix 
  3. Build the neural network and train the data 

The python file can be put into "..\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images" folder. 

The train accuracy reaches 99%, test accuracy reaches 96.5%  
Data source: http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset
