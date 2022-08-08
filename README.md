# improved-spork
Self-Supervised Learning with rotation prediction

[Final report](report.pdf)

Should download the following the dataset locally:
 - [Cityscape - leftImg8bit_trainvaltest](https://www.cityscapes-dataset.com/)
 - [COCO 2014 version](https://cocodataset.org)
 - [Labelled Faces inn the wild](http://vis-www.cs.umass.edu/lfw/)
  
 Can run for other datasets with creating a dataset class in [custom_dataset.py](custom_dataset.py)
 
 <br>
 
 Driver Code in
 - [main.ipynb](main.ipynb) is used for training models for the rotation prediction task.
 - [visualization.ipynb](visualization.ipynb) is used for observing the Saliency map and calculating other metrics.

Credits:
[solver.py](solver.py) code is adapted from the [CS 682 - Modern Introduction to Neural Network](https://compsci682-sp22.github.io/)

