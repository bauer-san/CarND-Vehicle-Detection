# Vehicle Detection #
Project 5 of Udacity SDC NanoDegree
----------
Geoff Bauer
4/9/2017 5:19:02 PM 
URL: https://github.com/bauer-san/CarND-Vehicle-Detection

----------
## The Project ##
The goals of this project include:
1. Write a software pipeline to detect vehicles in a video.
2. Write a detailed description of the project. 
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

---
# Project Writeup #

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg "car notcar"
[image1b]: ./output_images/HSV_car.jpg "HSV histogram - car"
[image1c]: ./output_images/HSV_nocar.jpg "HSV histogram - no car" 
[image2]: ./output_images/HOG_example.jpg "Histogram of Gradients"
[image2b]: ./output_images/normalized_features.jpg "feature normalization"
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/test_image1.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

---
### Explore the dataset(s) ###
The code for this step is contained in Section 1 of the `P5_classifier.ipynb` notebook.

I started by reading in all the `vehicle` and `non-vehicle` filenames and looked at a couple of random images form each class to get an idea of what data I will be working with.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

### Visualize and compare histograms of different colorspaces ###
I then explored different color spaces in Section 2 of the `P5_classifier.ipynb` notebook because I wanted to see what the histograms of different colorspaces looked like to see which colorspaces might work best to classify the images.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][image1b]
![alt text][image1c]

### Histogram of Oriented Gradients (HOG) ###
The code for this step is contained in Section 3 of the `P5_classifier.ipynb` notebook.

I started by visualizing the HOGs of each channel of several colorspaces.  I used the `skimage.hog()` function and values for the parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) that worked well for the course lessons.  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  A visualization of the HOG on a car in HSV colorspace HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` is shown here:
![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters. ####

I settled on the HOG parameters by trial and error with a goal of a good Accuracy score on the classification step and not be too slow.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them). ####

I combine the features in Section 4 , set the parameters in Section 5, and do the feature extraction in Section 6 of the `P5_classifier.ipynb` notebook.

In Section 7, I normalized the features to not bias the classifier to the higher valued features.  An example image and feature values before and after normalization are shown here: 
![alt text][image2b]

I randomized the order and split the datasets at the end of Section 7 and in Section 8, I trained a linear SVM using the features selected in Section 5.  The features and parameters were chosen by trial and error with a goal of a good Accuracy score on the classification step and not be too slow.

An Accuracy score of 98% was achieved using the parameters:
    cspace = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 2 # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16# Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [350, 650] # Min and max in y to search in slide_window()

### Sliding Window Search ###

In Section 9 I set up the window positions and scales to search.  Due to the perspective of the image, the vehicles farther away will be smaller but the smaller windows are computationally expensive.  So, I search small windows near the road horizon and larger windows from horizon to hood.  An example of the search windows is show here:
![alt text][image3]

Ultimately I searched on four scales using the features and parameters defined in Section 5.  Here is an example image:
![alt text][image4]

---

### Video Implementation ###

Here's a [link to my video result](./P5_project_video_out.mp4)

In the `process_image()` function, I recorded the positions of positive detections in each frame of the video in the `hot_windows` variable.  From the `hot_windows`, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle and overlaid boxes to cover the area of each blob detected.  With each new frame, the `heat` is decayed to 85% of the previous frame to provide some memory to the detection. 

---

### Discussion ###
I was surprised that the pipeline from the lessons worked reasonably well on the project video almost 'out of the box'.  It was necessary to use different window sizes to achieve a good result but the small windows also cause some false positives to 'leak through' the heatmap thresholding.

I am sure the pipeline would not work well, or maybe at all, in different light conditions.

If given more time, I would like to try adjusting the overlap of the small search windows and also process the project video using some of the other colorspaces.

### References ###
[Self-Driving Car Project Q&A | Vehicle Tracking](https://www.youtube.com/watch?v=P2zwrTM8ueA) 
[Reference implementation (Matt Zimmer)](https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5)
[GTI Vehicle Image Database](http://www.gti.ssr.upm.es/data/Vehicle_database.html)
[KITTI Project](http://www.cvlibs.net/datasets/kitti/)