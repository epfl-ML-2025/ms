# CS-233: Introduction to Machine Learning  **Milestone 2** 

Welcome to the project-MS2 for the course Introduction to Machine Learning. Our goal will be to implement a few of the methods seen in the course and to test them on a dataset. We will be evaluating them on evaluation metrics specific for the tasks and present our results with a project report at the end of each milestone. 

As deliverable, you are expected to fill a code framework: python files .py that implement these specific methods, and then a main python script main.py to call them on the data. We will use a subset of the [DermaMNIST](https://medmnist.com/) dataset, which is a small dataset of dermoscopic images of skin lesions across seven diagnostic categories: the goal is to output the correct diagnostic for each image.

You can find more information in the respective chapters below.

## Important Information:

* This will be a group project and must be done in **groups of 3\.** Please inform us if you are not 3\.  
* You may post your questions about the project on the "[Student Forum](https://edstem.org/eu/courses/1169/discussion/)" on Moodle.  
* *You must implement all the code by yourself*. You are not allowed to import and use external libraries (e.g., scikit-learn), unless it is *explicitly* said otherwise.  
  * NumPy, PyTorch, and matplotlib are fine to use.  
* **Deadline**:   
  * Milestone 2: 01.06.2025 (23:59)

## Grading

MS2 is also worth 10% of your final grade and will be graded as following: (**10pt**)

* Code (**3pt**): the framework code was filled as asked and works.  
* Clarity and presentation (**3pt**): the report is clear, well presented and structured.  
* Discussion (**4pt**): discussion of the work, what you did, what worked and didn’t, how you made your choices, reporting of the results, comparison of methods, etc.  
* *Creativity* (**1pt**): if you went beyond what is asked, e.g., by implementing something more, an interesting analysis or experiment, an especially well written code, etc.  
  * Note: while you can earn up to 11pt, each milestone will be capped at 10pt.

**Respect of the page limit:** We will remove 1pt if the report does not respect the page limit.

## The Framework

### Structure

The framework is organized with the following folder structure: (you can download it from Moodle)  
	**\<sciper1\>\_\<sciper2\>\_\<sciper3\>\_project/**  
main.py  
test\_ms2.py  
report.pdf  
**src/**  
	\_\_init\_\_.py  
data.py  
utils.py  
**methods/**  
	\_\_init\_\_.py  
	deep\_network.py  
dummy\_methods.py

* Please do not rename these files, nor move them around.   
* Your report will also be a part of this folder structure (as report.pdf).   
  Please include it in your framework when you are preparing your submissions. You will be submitting the zipped "project" folder.  
* You can add your dataset inside the project folder (you will download it from moodle), *but make sure you remove it for the submission*, otherwise the file size will get very large\!  
* Make sure to take a look into src/utils.py for useful functions you could use.

### Implementation

Some parts of the code are provided to help you get started. It will be your task to fill in the missing parts to get all the methods running on your data. You have to 1\) fill the python files implementing the methods and 2\) write the main script that runs these methods on the dataset.

* **Methods:** The methods are all implemented in the form of Python classes, under src/methods/. We have provided the implementation of a "dummy" classifier that returns random predictions as an example, in src/methods/dummy\_methods.py. Study this class well, you can use it as templates to code your own methods. It contains some essential functions:  
  1. \_\_init\_\_(self, ...)  
     * This function is used to initialize an object of the class. You can give it arguments that will be saved in the object, such as the value of hyperparameters.  
  2. fit(self, training\_data, training\_labels)  
     * This is the training or fitting step of the model. Training data with corresponding labels are used to estimate the parameters of the model. This method should be called before the predict function.  
  3. predict(self, test\_data)  
     * It generates predictions for the given data. It should be called after the fit function (you cannot predict without a trained model\!).  
* **Main script:** Contrary to the notebooks of the exercise sessions, we run python as a script in the terminal (see below for details on how to do it). You need to complete this script main.py to apply the methods to your data and evaluate them. Some pointers are given in the form of comments in the file. In brief, you are expected to follow these general steps:  
  1. Loading the data  
  2. Preparing / pre-processing the data  
  3. Initialize the selected method  
  4. Training and evaluating the selected method  
  5. Reporting the results  
* This script can be broken down in two main parts:  
  1. The main function def main(args): which is where we write the code we run.  
  2. And some code starting with if \_\_name\_\_ \== '\_\_main\_\_':. This is some pythonic way of coding “run the following instructions only if this file is a script”. This is where we parse the arguments that were given in the command line and then call the main() function.

### Running the Project

You will be running the Python scripts from your terminal. For example the main script can be launched from inside the project folder as:

python main.py \--data \<where you placed the data folder\> \--method dummy\_classifier

You can specify the method and many other arguments from the terminal by writing them as \--arg value in the command above. The arguments are defined at the bottom of the main.py script. We encourage you to check how they work, what their default values are, and you can even add your own\!

### Test Script

We also provide you with a test script test\_ms2.py. This script is for you to verify your project folder and code structure, as well as testing your method on some *very easy* cases.

* These easy cases are by far not exhaustive and you should verify the correctness of your code by your own means as well\!  
* The script can be launched from inside the project folder as   
  * python test\_ms2.py  
* By default, it hides any print() from your code. You can optionally pass it the argument \--no-hide to re-enable the printing.

## Milestone 2 

For Milestone 2 (MS2), you have to implement:

* Deep Networks with PyTorch  
  * MLP and CNN  
* The main script that calls them.

And you will also need to write a **2-pages report** about what you did.

* **Deliverable:** You need to submit **your code** and a **2-pages report** (zipped together). The project report should be part of your folder structure (see above\!)   
  * You should name the project folder (replacing the \<sciper\#\> by your own scipers) \<sciper1\>\_\<sciper2\>\_\<sciper3\>\_project, and the zipped file should have the name \<sciper1\>\_\<sciper2\>\_\<sciper3\>\_project.zip. Please make sure that when you unzip it, it extracts the folder \<sciper1\>\_\<sciper2\>\_\<sciper3\>\_project.   
* **Code:** You will need to complete the relevant parts of the following python files for this milestone. Please look inside the files for more information.  
  * main.py  
    * The main python script we use to run our methods. It works like the main.py script of MS1. You will need again to prepare the data, validation set, etc. then fill out the parts calling the deep network methods.   
  * src/methods/deep\_network.py  
    * There is 3 class to fill using PyTorch, two models and a trainer:  
      1. an MLP model, which should not use any convolutional layer and take vectors as input,  
      2. a CNN model, which should use at least one convolutional layer and take images as input,  
      3. and the Trainer class that is used to fit and predict with a deep network model.  
    * For the Trainer, you will need to complete the functions train\_one\_epoch() and predict\_torch(). We have already provided the functions fit() and predict() that serve as an interface between NumPy and PyTorch, so that we can call the trainer in the main script like the other methods from MS1.  
* **Data:** We will be using the DermaMNIST dataset for MS2.   
* **Report:** You need to write a concise *2-pages* report about your work. The project report should include your methodology and a brief summary of the results you have achieved with the methods.   
  * We advise to follow a structure like “Introduction, Method, Experiment/Results, and Discussion/Conclusion”.  
  * In Methods, you should explain your data preparation pipeline, what was your process to select hyper-parameters, describe your models and what you tried (layers, dimensions, …), and anything special you may have done.  
  * For the results, how do the models perform with respect to hyper-parameters and architecture choices? You should also report the speed of the training/inference.  
    * You can, and probably should, use graphs and/or tables.  
  * Finally, discuss what you obtained, explain any surprising results you may have had, which method is best and why you think it is so.  
    * Compare your final models (MLP, CNN) on the *test* data.  
  * Stay concise and respect the page limit. You can use double column if needed.  
* **Test:** Make sure that test\_ms2.py runs without any problems\! This ensures that your code compiles and your implementations can be imported.  
  * And make sure that your project is running when you call main.py for each of the methods.The following commands should run without crashing:

**Some additional points:**

* Make sure that your method classes **do not modify the data** they are passed. If you are doing some form of data augmentation (such as adding a bias term, doing feature expansion etc.), then you must do so *outside* these classes, in your main.py script.  
* Make sure your implementations of these methods are not dataset specific–they should work for arbitrary numbers of samples, dimensions, classes etc.  
* Make sure that your project is running when you call main.py for each of the methods. In other words, the following commands should be running without crashing.

**Sample commands to run:**

CNN (with lr=0.00001 and max\_iters=10 in that example)  
python main.py \--data \<where you placed the data folder\> \--nn\_type **cnn** \--lr 1e-5 \--max\_iters 10

MLP (with lr=0.00001 and max\_iters=100 in that example)  
python main.py \--data \<where you placed the data folder\> \--nn\_type mlp \--lr 1e-5 \--max\_iters 100

### MS2 Competition

* We are running a competition where you are supposed to upload your classification predictions from your best performing neural network model on the test data. Your ranking will not determine your grade, but it is mandatory to submit at least one attempt for each group. The top performers will earn a chocolate gift from us.   
* This is the reason why we did not release the test labels in the dataset on Moodle. More details will be updated here this week. 

You can add the \--test argument to predict on the actual test set, otherwise your code should use a validation set\!

### Sample results

For the deep networks, we won’t provide sample results; try to get the best performance you can with either model. 

## The DermaMNIST dataset (For MS2)

The DermaMNIST dataset contains dermoscopic images of 7 categories of skin lesions. Derived from the HAM10000 dataset, DermaMNIST serves as a standardized collection of dermatological images for machine learning applications in medical diagnosis. The dataset consists of 9,012 images divided into a training set and a test set, with each image resized to 28×28 pixels, with 3 color channels. Below is the number of images for each class.

| Label | Description | Count |
| ----- | ----- | ----- |
| 0 | Actinic keratosis | 294 |
| 1 | Basal cell carcinoma | 462 |
| 2 | Benign keratosis | 989 |
| 3 | Dermatofibroma | 103 |
| 4 | Melanoma | 1,002 |
| 5 | Melanocytic nevus | 6,034 |
| 6 | Vascular lesion | 128 |

The dataset is split into 7,007 train images and 2,005 test images. The data.py file contains a helper function load\_data() to load the images and the corresponding labels (for classification) of the skin lesions. The labels have been cast into integers in {0,1,...,6}. The function returns as np.array:

* train\_images of shape (7007, 28, 28, 3\)  
* test\_images of shape (2005, 28, 28, 3\)  
* train\_labels of shape (7007,)  
* test\_labels of shape (2005,)

## The Tasks and Metrics

To measure the performance of our methods, we can use the following metrics.

### Classification Metrics

#### Accuracy

We will report the average accuracy in percentage. It can be written as:  
![][image1]  
Generally speaking, in case of imbalanced datasets (meaning that the dataset is not divided evenly among the different class labels), the accuracy metric can be too biased toward the largest classes.

Therefore, we will also report the F1-score, which may be more meaningful than the accuracy metric in these cases.

#### F1-Score

We will report the **macro F1-score**, which is an average of class-wise [F1-scores](https://en.wikipedia.org/wiki/F-score). 

The F1-score for a class *i* is defined as follow:  
![][image2]  
where 	![][image3] and ![][image4] are the precision and recall values for the class *i*. Such precision and recall are computed as  
![][image5]

* *True Positive*: true class is *i* and predicted class is *i*  
* *True Negative*: true class is *not i* and predicted class is *not i*  
* *False Positive*: true class is *not i* and predicted class is *i*  
* *False Negative*: true class is *i* and predicted class is *not i*

Precision relates to how many of our predictions as class *i* are correct, while Recall relates to how many of the true samples of class *i* were predicted as such.

For K classes, we can compute a macro F1-score as:  
![][image6]  
Higher F1-scores correspond to a better model and macro F1-score reflects the model’s average performance on all the classes. 

### Runtime Analysis

It is generally very useful to report how fast your algorithms can be trained and how fast they predict. You can easily measure the time passed in Python using the time module. Here is a quick example:

import time  
s1 \= time.time()  
dummy\_function()  
s2 \= time.time()  
print("Dummy function takes", s2-s1, "seconds")

In case you get an error for import cv2   
Install:  
pip3 install opencv-python==4.6.0.66  

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJ4AAAAqCAYAAABC3uIYAAAFQklEQVR4Xu2cW8htUxTHh8sRkcSLUhxKpFziIJJbeVJKRBQR5UkO5ZwH0s7lTV548UCK41aUvKDkuMslPEhyaYUj14hT7pf5b65hz/3fc8w11/r2Xt/ea89fjb6+/5hr7bHnGN+6zLXGJ1Loi7NZWAHOZKHQL+87O5fFFeACZy+zuAo86uzf2t5ztvukuxdul8nJHzl7TsZxbXN2n7Mnnf1Za+fo4HVgJJPxPSI+xrcD7XwdnAG++3UsDplfnB0Z/P6D+EnbEGjz5hnxnxlDk8jcIF6/lB09Y8V3mnj9LXYkiO1nkBwg/svuFmi71tp3gTZv8HmPs1hjJRakfH2RiiHli/GUswdYHCJXiZ+YF0jPmTCcUg5lMeAEZ4+xGGGLsx+d7cKOmlQsKV8Tn7JAfMaCQSoG9eVeuuiBYCW419mepKUmU9lb/JjP2eHYJN6Xc7reKf76zsKKRT//e3Zksl3s2PGH8AQ7DKz4APS/WGzA2tfguVz8l7+C9Bj7OHvJ2ReBpkWX+1eOsYewGBBLLI600H4jvS1W7LlFBzQ+HK1gJzn7uNa63Kliu7NYXAXwxT9gsYFXnH0p7YsOYLx1mgWaWFwDqt0s8WLdQ6aLtIm1xA40vo21XRJoXWKED99vpfhZ2hed8pF0T1wKTWIOOLV/zWIGXWMHsfiwH90n0xTjr87uYHHIvCP+rqoLp4if5DfEHz3aEEtOSCyxs2QtsQMrvqvF6xexowFsg6WileAhZ3eTFpvMGCeLH6tLMq852zF2N4Jt92MxwEosc7jkj1Ws2LGvXKzPvFC8zstE1ngFvvVcGO+N65097ezo2o4R/9gqZznhJ5lMnIIEfkWaBba/hsWApkQxuWNPFDt26EeQbmHFp9eMvwca5hrExivw7cvi0DhOxhPHdmcwLkbl7GGZTpyCBL7IYgTsA2uCzEj8kVjjwWO9G8MBBqmkhsSKTkHim/Yzksn4HnR2Szgg8IHbnB0Y6BZTvmdlfMEIw6RCg70b6KfrBgF3iX+uV5gGa4iYt2vZ0ZGpxK0zW5296uyyQLNivEkSZ5qwipk3ZVz1uLPBOCwg4uf2Wi9Mg9ORNadtmdV+5okVI3SsBUZJFR54ngUphdcEnkLgjhIX5F053tkfzr6RtS8szxMrxiudfUjaBLHCw0YKdsyUwssD85TzmG1o6OO/JLHC49+ZUnj5xJ6fDp1vWYihhXesszPEn1rnXXh4VemellYYGFp4F4tff9IHwilShXcwCwZYVGxjhYHBp9q96PcYqcL7x9lBLBYKDBeeailShYe7uf1Z7AEcFW8tttB2ngTECq+JVOHlgBX2+1taYWDMsvC2iffldCOd2tIKA6Nr4X3CYg3e/c8pvMKKgtVmrLfsEH9dhpf5dk6MmAZ+jMN4bIft8UbHUcEY3BWXwiv0zlALbyTL25D9d60NMS//g1Mw/oXBULEuTWbdkH0YC5lY8XVpyF4a8Ko52vNwSt5AvqFgJRakfG2pWMgkFUPKV1hwUslL+dpSsZBJKgb14RW3wpJhJXatDdlMxUImVnwAetuG7MKCEEvsrBqyQyoWMtH4wobszbXWpSG7sCBoYufVkK1ULGSi8W2sDQ3Zr9farGMs9EjsiGeBG6xUszPAUVL3mWNN+7Pi69qQXVgQrMTOmoqFTKz4ujZkFxYEK7Ex2oxlKhYysT4T/R/Q2zZkFxaEtolqMzakYiETK75N4vW2DdmFdWYk823IZioWGhjJfBqyC0tK16RWLMyINg3ZhSWma1Jz/u3trOgaY2FBQU9yrNl5keg1xv8AEGMvIjXgZbwAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAPCAYAAAA71pVKAAAAqklEQVR4XmNgYGBYCsQ/gPg/ED8G4rlAPA+IT0DFLjAQAM4MEIV26BIMEHEQxgmcGCAKbNElgGAlA0TODV0CBvBpnsAAkatHl4ABfJq/MkDkWNElYACX5llQ8Qg0cRQA03wZiFdB8WIgDkJWhAQ0kTkwzZHIgjiAIQNa6JOiGQOQolkJiD2RBUjRrMyA5Ow+IH4PFfjCAIlXRpgkDoDiZ1LBwGg2Y4BodgcAKBEu04h1bNIAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAPCAYAAADtc08vAAAAzklEQVR4XmNgYGBYCsQ/gPg/EL8F4vlAvAGI/0LFAhmIAM4MEMV2aOI2UPHTaOIYwIkBotAWXYIBIg7CeAExBrCgSyADQgb8QRdEBzADAoBYGIgVgLgAKnYYoQw3gBkQxgDRbADEx6Fi8ghlKEAAmYPLCzeh4uhgIwOaOC4DUqDiIJfhBbgMCIGKr0IT1wHiIGQBmAHoCckEKv4TSawQiB2h4mDQB8TvoQLvoHxkABKHKW4GYgkkcaJBORAfBeJYJDGSDMAGKDLAkwFigCEA9aE7JfRKqesAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,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>