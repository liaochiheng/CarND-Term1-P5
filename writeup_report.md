**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/bbox.png
[image4]: ./output_images/sliding_window_frames.png
[image5]: ./output_images/bboxes.png
[image6]: ./output_images/heatmap.png
[image7]: ./output_images/labeled_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell(Utilitity functions for features extraction) of the IPython notebook [P5.ipynb](https://github.com/liaochiheng/CarND-Term1-P5/blob/master/P5.ipynb).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried combination of Spatial_bin + Color_hist + Hog_all, and that worked fine.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I defined a class of classifier `CarSVC` in code cell 3(Define a classifier for car detection).

In the `train` function, i split the train and test data manually:
* Use 'datasets/vehicles/KITTI*/*.png' for car train data
* Use 'datasets/vehicles/GTI*/*.png' for car test data
* Use 'datasets/non-vehicles/Extras/*.png' for not-car train data
* Use 'datasets/non-vehicles/GTI/*.png' for not-car test data
* Split both car test and not-car test data, 20% into train-data, 80% left remain to test data.
* The code is:
```python
rand_state = np.random.randint(0, 100)
X_train1, X_test, y_train1, y_test = train_test_split(
X_test, y_test, test_size=0.8, random_state=rand_state)
X_train = np.concatenate( (X_train, X_train1) )
y_train = np.concatenate( (y_train, y_train1) )
```
   
After that, the SVC classifier worked better than before, and reduced some overfitting.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In code cell (Define a CarFinder class to find cars), I defined a class `CarFinder` for car-detection, and in function `CarFinder.find_cars`, I implemented a sliding window with `orient=9, pix_per_cell=(8*8), cell_per_block=(2*2), scale=1.5, window=(64*64)`. That works just ok.

Here is a demonstration with detected bound boxes drawing on:
![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Optimization the performance of the classifier have been mentioned above, with splitting datasets manually.

Ultimately I am using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

I used both scale=2.0 and scale=1.5 to search cars.
* For scale=2.0, is more confident, there must be a car.
* For scale=1.5, is less confident, maybe a car, maybe a false positive.
* For each scale, I added different values into heatmap: `CarFinder.heat_thresh:`
```python
def heat_thresh(self, img, thresh=4):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    bboxs = self.bboxs[-6:]
    for bbox1, bbox2 in bboxs:
    for box in bbox1: # More confident
    heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += thresh + 1
    for box in bbox2: # Less confident
    heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    heat[ heat <= thresh ] = 0
    return heat
```

Here are some example images on consecutive 6 frames:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding bboxs:

![alt text][image5]

### Here are six frames and their corresponding heatmap:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1). In some frames, the classifier failed to find any cars. Maybe i need to keep cars tracking.
2). The train and test data are not good enough to train a good classifier, even after manually splitting.

2. Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

1). I manually splitted the train data, kind of reducing some overfitting.
2). I tried two scales(2.0 and 1.5), which works better than before.
3). I didn't do tracking, which will make connections between ajacent frames, and that will make a better detection. I will try this as well.

