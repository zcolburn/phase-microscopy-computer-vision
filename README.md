# Computer vision classification of phase microscopy images of scratch wounds
This is a computer vision model for classifying regions of phase microscopy images of scratch wounds as part of the cell sheet, wound, or wound edge.


## Organization of the data
Data should be stored in a folder specified by the **directory** parameter in *Main.py*. Within the *Data* directory, folders should be organized in the following hierarchy:

* Experiment name
  * Condition
    * The number/name of the well with the scratch
      * The number/name of location within the well
        * Data files

There should be two types of data files in the lowest level directory. Specifically, there should be one *tif* file of the form '*FileNameHere.tif*'. This file should be a time series with a single channel. The remaining files should be of the form '*FileNameHere_0m.txt*' where 0m indicates time 0 minutes. '*_30m*' would indicate 30 minutes. These outline files should have the x and y coordinates in pixels of the wound edge. These outlines were made in ImageJ by outlining everything to the left of the scratch wound. Using the distance in pixels of the parameter **bumper**, specified in *Functions.py*, coordinates near the edge of image are excluded. Thus, the wound edge coordinates can be identified. Everything to the left of these coordinates is cell sheet, everything to the right is wound.


## Training the model
The model can be trained by running *Main.py*. This will result in the images in the **directory** specified to be converted into numpy arrays that will be used for training.


The model specified in *Main.py* is summarized in the file *ModelSummary.txt*.


## Model accuracy
After a week of training on a desktop computer (4-core, i7, 2.4 GHz), the model reached an accuracy of 93.5%.
