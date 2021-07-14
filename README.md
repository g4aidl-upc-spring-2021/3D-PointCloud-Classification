# AIDL Spring 2021: 3D Point Cloud Classification

This repository contains two neural network architectures applied to point cloud classification. The first one follows the implementation proposed in the paper [_PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation_](https://arxiv.org/pdf/1612.00593.pdf). The second one is defined with no reference and is based on Graph Convolutional Networks (GCN).

The original goal of the project aimed to perform a part segmentation task over point clouds structured data, but due to development difficulties related to time-execution and memory in terms of storage capacity, we decided to switch and simplify the task to classification. Thus, our proposed solution focuses on the effectiveness of these two architectures in the predictive classification performance on point clouds retrieved from [ModelNet](https://modelnet.cs.princeton.edu/) open dataset.

### About

Final Project for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310400/postgraduate-course-artificial-intelligence-deep-learning/) 2020-2021 spring online edition, authored by:

* [Berta Núñez de Arenas Millán](https://www.linkedin.com/in/berta-n%C3%BA%C3%B1ez-de-arenas-mill%C3%A1n-0b9993150/)
* [Eduard Salagran Ferragut](https://www.linkedin.com/in/eduard-salagran-ferragut-83a118162/9)
* [Juan Vigueras Díaz](https://www.linkedin.com/in/juan-v-012124180/)

Advised by [Albert Mosella-Montoro](https://www.linkedin.com/in/albertmosellamontoro/)

## Repository Distribution

The distribution of the project files in this repository is as follows:

	├── src          # Source files
		├── Models
			├── PointNet.py
			└──GCN.py
		├── main.py
		├── utils.py
		└── dataset.py
	├── collabs      # Collabs to execute the code and the experiments easily
	├── checkpoints  # Trained models
	├── .gitignore	
	└── README.md
		
## Instructions to Run the Code


### Python

#### Installation with Conda

Create a conda environment by running

```python
conda create --name give_a_name
```

Then, activate the environment

```python
conda activate give_a_name
```

and install the dependencies

```python
pip install -r requirements.txt
```

#### Execute the project

To run the project with default configuration, run

```python
python src/main.py
```

To run the project with custom configuration, run 

```python
python src/main.py [--batchSize=<bs>] [--debug=<db>] [--epochs=<e>] [--dataAugmentation=<da>] [--normalizeScale=<ns>]
  [--numOfPoints=<np>] [--flipProbability=<fp>] [--flipAxis=<fa>] [--rotateDegrees=<rd>] [--rotateAxis=<ra>]
  [--model=<mod>] [--numFeatures=<k>] [--numClasses=<nc>] [--level=<lev>]  [--dropout=<do>] [--optimizer=<opt>]
  [--learningRate=<lr>] [--weightDecay=<wd>] [--momentum=<mom>] [--schedule=<sch>] [--gamma=<gam>] [--patience=<p>]
  [--stepSize=<ss>]
```

### Collabs

You can execute the whole project in a row by opening it in Google Colab and running all the cells.

***

## Table of Contents

- [1. Introduction](#intro)
	* [1.1. Motivation](#motiv)
	* [1.2. Objectives](#obj)
- [2. Corpora](#corp)
	* [2.1. Data Description](#ddesc)
	* [2.2. Data Pre-Processing](#dprep)
- [3. Implemented Architectures](#arch)
	* [3.1. PointNet](#pointnet)
	* [3.2. Graph Convolutional Networks](#gcn)
- [4. Experiments Logbook](#exp)
	* [4.1. Evaluation Metric](#metric)
	* [4.2. Experiments with PointNet](#exppointnet)
	* [4.3. Experiments with Graph Convolutional Network](#expgcn) 
- [5. Conclusions](#conc)
	* [5.1. Experiments Conclusions](#expconc)
  * [5.2. Project Conclusions](#proconc)
- [6. Achieved Milestones](#milestones)
- [7. Further Work](#fwork)
- [8. References](#ref)

***

<a name="intro"></a>
## 1. Introduction

<a name="motiv"></a>
### 1.1. Motivation

Classification tasks serve as a foundation for solving higher level problems such as object recognition, interaction analysis and scene understanding. Point clouds are a simple and unified structure that can solve these problems. Furthermore, point clouds can either be directly rendered and inspected or converted into models using various shapes and patterns. 

Some example of scenarios in which point clouds are used as the offspring reality representation from scanner / 3D cameras are robotic perception, augmented reality, drone and marine photogrammetry and surveying among many others, as you can see in the following images:

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/motiv.jpg "Point clouds retrieval applications")

<a name="obj"></a>
### 1.2. Objectives

The main purpose of the project is to demonstrate the effectiveness of these two types of architectures in point cloud segmentation task. In order to address this, the scope can be split into the following sub-objectives:
* Learn to work with 3D point clouds such as [ModelNet](https://modelnet.cs.princeton.edu/) dataset and use them in Deep Learning. This includes exploring, cleaning and preprocessing the data to make it suitable for each type of network architecture.
* Reproduce a scientific publication’s network architecture such as [PointNet](https://arxiv.org/pdf/1612.00593.pdf) from scratch.
* Develop and implement a classifier network architecture based on Graph Convolutional Networks. 
* Train these two classifier models in order to recognise ten different classes of objects. This includes undertaking several tests in order to fine tune the models and obtain the best configuration of each of them.
* Carry out an analysis of the results with suitable metrics such as mean accuracy and improve baseline network applying according steps depending on its performance and results.
* Draw conclusions from all the tackled experiments and the different attempted improvements. Define further steps for the project based on its closure.

<a name="corp"></a>
## 2. Corpora

<a name="ddesc"></a>
### 2.1. Data Description

For training and testing our models, we have used Princeton’s [ModelNet](https://modelnet.cs.princeton.edu/) dataset. ModelNet includes three-dimensional mesh structure shapes with a view to enable research in computer graphics, computer vision, and other related disciplines with a comprehensive clean collection of 3D computer-aided design (CAD) models for objects. Further information about its conception and design can be consulted at the [original paper](https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf).

ModelNet is very easy to download and implement from [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#) as a class, so data can directly be retrieved from it without the need of creating an additional Dataset class from scratch.

For the sake of simplicity, the version of ModelNet used in this project is ModelNet10, which includes ten types of popular objects categories that are divided in Train and Test splits. In order to evaluate the models during training, a random sample of 20% train objects has been taken as a Validation split. The final distribution is the following:

| Train: 3193 | Validation: 798 | Test: 908 | 

As mentioned before, the dataset consists of 10 different types of popular objects, but the frequency of these objects is not balanced and equally distributed, i.e., there are some objects that have more point clouds than others. In these histograms we can see the frequency of these objects in each one of the splits:

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/dataset.png "Dataset Frequency")

This imbalance is not considered along the project when applying optimization techniques.

<a name="dprep"></a>
### 2.2. Data Pre-processing

#### Converting meshes to point clouds

Data objects from ModelNet10 hold mesh faces instead of edge indices as in graphs. In order to convert these meshes into point clouds to be able to handle for PointNet architecture, the transform method [SamplePoints](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html?highlight=samplepoints#torch_geometric.transforms.SamplePoints) from Pytorch Geometric is used. This samples a fixed number of points from the mesh faces according to their face area. In this case, the chosen amount of points to sample is 1024, as it is proposed in the reference paper.

#### Converting meshes to point clouds with edges (graphs)

In order to convert meshes into graphs to be able to handle for GCN architecture, the transform method [KNNGraph](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html?highlight=KNNGraph#torch_geometric.transforms.KNNGraph) from Pytorch Geometric is used. This applies the k-nearest neighbor algorithm in order to create a graph by using points as node positions.

#### Data Augmentation

In order to perform data augmentation techniques when fine adjusting the models, applied transformations from Pytorch Geometric include [Random Flip](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html?highlight=random%20flip#torch_geometric.transforms.RandomFlip) and [Random Rotate](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html?highlight=random%20flip#torch_geometric.transforms.RandomRotate). 

These transformations are applied as a regularization technique with the purpose of reducing overfitting as applicable. They are very simple to use, only indicating the transformation when loading the dataset they will automatically be applied when retrieving the data at the training loop. 

As the names imply, _RandomFlip_ flips node positions along a given axis randomly with a given probability; while _RandomRotate_ rotates node positions around a specific axis by a randomly sampled factor within a given interval of degrees. 

<a name="arch"></a>
## 3. Implemented Architectures

<a name="pointnet"></a>
### 3.1. PointNet

PointNet is a network that accepts point clouds as input and it can be applied to classification, part segmentation or semantic segmentation tasks. Focusing on the objective of this project, the network just when applied to object classification will be explained. A diagram of the architecture would be as follows:

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/pointnet.png "PointNet")

The input of the PointNet takes one cloud of N points at a time - this N might change depending on the point cloud if the number is not fixed when pre-processing. Each of these N points have three features - in this case, cartesian coordinates _x, y, z_. When it comes to working with three dimensional points, there are two main constraints:

* The network must also be invariant to rigid transformations such as rotation or translation.

To ensure this, the input is passed through a module called _input transform_. In order to align extracted point features after applying a fully connected layer, ._feature transform_ is applied hereunder.

* Point clouds are unordered. The network has to be invariant to permutations of the input.

The TNet applied in both transforms is composed of basic modules of point independent feature extraction, max pooling and fully connected layers. Intermediate fully connected layers between transforms are used for learning spatial and feature encoding for each point and mapping them to a higher-dimensional space.

After input and feature transforms, the network applies a _max. pooling layer_  as a symmetric function to the extracted and transformed features so the result does not depend on the order of input points. At this step, every point feature is aware of both local and global information

The last step is a fully connected layer that returns the _k_ output scores - in this case, _k_ corresponds to each of the ten classes of ModelNet.

<a name="gcn"></a>
### 3.2 Graph Convolutional Network

Graph convolutional networks are an extension of graph neural networks with the exception that convolutions are used in order to be able to work with both nodes and edges of the graph. Layers applied in this architecture are inspired in the paper [_Semi-supervised classification with graph convolutional networks_](https://arxiv.org/pdf/1609.02907.pdf) and can be easily implemented through the [Pytorch Geometric corresponding class](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv).

An attempt of implementation was made by defining an architecture made of one, two and three sequential graph convolutional layers each followed by a batch normalization and a ReLU activation. 

At the end of GCN layers and similarly to PointNet, a _global max. pool_ followed by a fully connected layer are attached to finally obtain the _k_ scores for each one of the classes. 

A diagram of the proposed architecture with three layers would be as follows:

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/gcn.png "GCN")

The number of channels is chosen as typical values for basic convolutional architectures and the number of points at the input of the network is chosen to be the same as in the PointNet network.

<a name="exp"></a>
## 4. Experiments Logbook

The designed and performed experiments are presented in this section in chronological order of execution. They are separated between PointNet and GCN experiments and details from each of them are explained as follows:

* How is the network designed
* The modifications made and thus the difference from the previous experiments
* The results obtained and the conclusions for each one, justifying which experiments are concluded to be the best in terms of overall performance and evaluation metric

<a name="metric"></a>
### 4.1. Evaluation Metric

The main criterion to select the best model of each set of experiments is based on the best validation metric obtained in each training. In this case, and following the choice of referenced paper, __accuracy__ is chosen. It is very easy to implement from [TorchMetric library as a class](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#accuracy), and it is defined as:

![equation](https://latex.codecogs.com/gif.latex?accuracy&space;=&space;\frac{1}{N}\sum_{i}^{N}1\left(y_{i}=\hat{y}_{i}\right))

Accuracy is computed at each epoch after data of predictions and targets are added to accuracy class at each iteration of the batch. After each epoch, metric class is rebooted.

If relevant, other factors might be taken into account when determining the best performance, like overfitting presence.

<a name="exppointnet"></a>
### 4.2. Experiments with PointNet

In the experiments using PointNet architecture the major goal is to improve the validation accuracy. So as to achieve this, several features will be sequentially added - starting with a basic network, without normalising the data and without applying data augmentation or other regularization techniques. 

All experiments share some common characteristics: 

* ModelNet10 dataset is used.
* Number of epochs during training is set to 100.

As well as some hyper-parameters:

* Batch size: 32
* Learning rate: 1e-3
* Number of workers: 2

While some other features are changed throughout the experiments for the sake of finding the optimal configuration or the best performance of the models:

* Optimizer: 
	Adam: weight decay of 1e-3 (as proposed in the reference paper)
	SGD: momentum of 0.9

* Scheduler (if applicable):
 	Step: divide learning rate by 2 each 20 epochs (as proposed in the reference paper)
	One Cycle: max. learning rate of 1e-3 in 100 epochs

* Data Augmentation (if applicable): 
	Random Flip: probability of 0.5 to be flipped at Y axis (as proposed in the reference paper)
	Random Rotation: max. degree of 45 around X axis (as proposed in the reference paper)
	Both of them

When and how to apply or change each one of these features is explained in the applicable situation.

#### Experiment 1: No normalization of coordinates

**Setup and hypothesis**

The first experiment using PointNet uses _raw_ data from ModelNet10 sampled at 1024 points per cloud. The optimizer that is used in this experiment is Adam. 
Before running this experiment the expected result is having a low validation accuracy and a certain level of overfitting, as no transformation is yet applied to the data. 
Since this is the most basic experiment, no very good results are expected.

**Results obtained**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp1PointNet.png "Accuracy and Loss")

ID | Best Val Accuracy
:-------------: | :-------------: 
1 | 0.74185 

**Findings**

The validation accuracy is low as expected in the hypothesis. As it can be seen in the chart, the network seems not to be generalizing well - that means there is a presence of overfitting since train accuracy has a reasonably good result while validation accuracy is much lower. As a well-known method to help neural networks improve their learning, normalization of data will be applied in order to expect better results of generalization.

#### Experiment 2: Normalization of coordinates

**Setup and hypothesis**

The conclusions of Experiment 1 were very clear, so in this experiment the same setup is used but also including the normalization of coordinates in order to scale each point cloud to interval (-1,1). 

The transformation used to normalize the points is [NormalizeScale() from Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html?highlight=NormalizeScale#torch_geometric.transforms.NormalizeScale) and it is applied to all three splits (train, validation and test). An improvement of validation accuracy is expected, but as dropout or data augmentation is not being applied yet (as they did in the original paper) overfitting is expected to appear.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp2PointNet.png "Accuracy and Loss")

ID | Normalization | Best Val Accuracy  
:-------------: | :-------------: | :-------------: 
2 | Yes | 0.95614 

**Findings**

Results are much better. The validation accuracy shows a very good result, but overfitting can be identified as it was expected: the training accuracy continues growing while the validation stops improving.

#### Experiment 3: Introducing Dropout

**Setup and hypothesis**

In order to try avoiding the overfitting found in the previous experiment, a regularization technique is being applied. As proposed by the original paper, dropout with probability of 0.3 is implemented in the second last fully connected layer of the architecture. By applying dropout in the network, reducing overfitting is expected, thus obtaining a better validation accuracy. 

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp3PointNet.png "Accuracy and Loss")

ID | Normalization | Dropout | Best Val Accuracy
:-------------: | :-------------:  | :-------------: | :-------------: 
3 | Yes | Yes, p=0.3 | 0.95614

**Findings**

The validation accuracy is as good as in the previous experiment, but plots are smoother. Since including dropout has not decreased the accuracy, it will be kept in further experiments. However, there is still overfitting, so two transformations for data augmentation will be applied: random flip and random rotate, trying them separately and then together.

#### Experiment 4: Introducing RandomFlip

**Setup and hypothesis**

This experiment is the first one to include data augmentation. The transformation that will be applied is Random Flip. The aim of using data augmentation is trying to reduce the overfitting, as only applying dropout is not enough. The main expected result from this experiment is removing the overfitting and having a similar validation accuracy as the previous experiments. 

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp4PointNet.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: 
4 | Yes | Yes, p=0.3 | Random Flip | 0.96241 

**Findings**

As it can be seen in the plots, the main hypothesis has been accomplished: the overfitting has been reduced a lot due to the data augmentation transformation applied. It can also be appreciated in the results that the validation accuracy has increased, achieving the best result so far. Hence, using the data augmentation might be useful as an extra feature to possibly avoid an extra amount of overfitting.

#### Experiment 5: Introducing Random Rotate

**Setup and hypothesis**

Since the first data augmentation transform has been very useful, the next step is to compare and see which is the one that gets better results. The second applied transform is random rotation. In this experiment Random Flip is NOT being used. As obtained in the previous experiment, similar results are expected: avoiding overfitting from experiment 3 and achieving a  similar validation accuracy.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp5PointNet.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Best Val Accuracy  
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: 
5 | Yes | Yes, p=0.3 | Random Rot. | 0.95488

**Findings**

Looking at the plots it can be seen that, similarly to experiment 4, overfitting has been reduced. The objective of applying data augmentation was reducing overfitting, but having a better validation accuracy was also expected. In this experiment the accuracy is worse than in experiment 3, in which data augmentation was not applied. Thus, Random Flip performs better than Random Rotation. The reason for this behaviour could be the two coordinates transformation from random rotation against the single coordinate transformation from flip rotation, which might imply a too strong alteration of data to let the network generalize a type of object.

#### Experiment 6: Introducing Random Flip and Random Rotation

**Setup and hypothesis**

Applying random flip to the data has improved the results and has reduced overfitting, while applying random rotation has decreased the validation accuracy. In this experiment both transforms are applied in order to see if, when combined, results improve. As in the previous experiments, reducing overfitting from experiment 3 is expected, as well as getting a better validation accuracy than experiment 4 (where we used random flip).

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp6PointNet.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: 
6 | Yes | Yes, p=0.3 | Random Flip+Rot. | 0.95865

**Findings**

The plots show how the overfitting has been mostly removed in this experiment. Applying the two data augmentation techniques has increased the performance of the network and has accomplished the pursuited objective when applying data augmentation. However, the validation accuracy is lower than in experiment 4. Applying only random flip transformation gives the best validation accuracy. 

#### Optimizer and learning rate

During the previous experiments the model to achieve best validation accuracy was found when applying:

* Adam optimizer

* Dropout

* Data augmentation

From now on, the main goal will be trying to find the best configuration of the optimizer (Adam, used in the paper or SGD to compare results) and learning rate using two different schedulers: the one used in the paper (LR Step) and another one to compare results (OneCycle). 

In these next experiments, the setup of experiments 4 and experiments 6 will be used. Experiment 4 gives the best accuracy, but it still has overfitting, while experiment 6 gives a lower accuracy but it doesn’t have overfitting. 

Applying the schedulers on both models will allow to see if a model with the highest validation accuracy is achieved without having overfitting.

#### Experiment 7: Random Flip + LR Step

**Setup and hypothesis**

Using a scheduler during the training will be the first step. In this experiment the scheduler used is LR step, which decreases by 2 the learning rate every 20 epochs. The model used is the one from experiment 4 which has the random flip transformation. This is the model with higher validation accuracy. The expected result from this experiment is increasing the validation accuracy as the scheduler is not used to reduce overfitting.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp7PointNet.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: 
7 | Yes | Yes, p=0.3 | Random Flip | LR Step | 0.96867

**Findings**

Applying the scheduler to optimize the learning rate improves the performance, achieving the best validation accuracy when compared with previous experiments. The hypothesis of this experiment was correct, since overfitting can be still appreciated in this model.

#### Experiment 8: Random Flip + Random Rotation + LR Step

**Setup and hypothesis**

As explained before, the scheduler in the model will be tried with only random flip and with both random flip and random rotation. This experiment will be the one that applies both transformations and the scheduler. Like in the previous experiment, the scheduler will reduce the learning rate by 2 every 20 epochs. Having a model with a higher validation accuracy than the previous experiment and without overfitting is expected to happen, as in experiment 6.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp8PointNet.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: 
8 | Yes | Yes, p=0.3 | Random Fip + Rot. | LR Step | 0.96616

**Findings**

A better validation accuracy than the one achieved was expected, as it is lower than the accuracy from the previous experiment. The overfitting, as supposed, is smaller than the experiment 7. It can be seen the effect of the scheduler as the validation has increased compared to experiment 6, but it is not as good as desired from expectations.

#### Experiment 9: Random Flip + OneCycle

**Setup and hypothesis**

Up until this point, the performance of the scheduler used in the reference paper has been tested, so a different one will be tried in order to compare results. In the following experiments the used scheduler is [OneCycle](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR). This scheduler anneals the learning rate from an initial LR to some maximum LR and then from that maximum to some minimum LR much lower than the initial one. This experiment is used with the model of experiment 4 (using random flip transformation) and similar results to experiment 7 (where the same model with LR Step is used) are expected - Better validation accuracy, but a model with overfitting.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp9PointNet.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: 
9 | Yes | Yes, p=0.3 | Random Flip. | OneCycle | 0.96491

**Findings**

A similar performance than experiment 7 was expected, but the validation accuracy is lower. It definitely won’t be the final model as the LR Step scheduler is better using the same model.

#### Experiment 10: Random Flip + Random Rotation + OneCycle

**Setup and hypothesis**

Like it was done with the LR Step scheduler, OneCycle is being tested with the model where random flip and random rotation transforms are applied. The expected results are similar to experiment 8.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp10PointNet.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: 
10 | Yes | Yes, p=0.3 | Random Fip + Rot. | OneCycle | 0.96992

**Findings**

Contrary to the previous experiment, in this model the OneCycle scheduler improves the validation accuracy, achieving once again the best result so far. Using this model configuration with OneCycle scheduler gets the higher validation accuracy and a model with less overfitting than previous ones.

#### Experiment 11: Random Flip + Random Rotation + SGD Optimizer

**Setup and hypothesis**

The configuration with the best optimizer is being pursuited, some experiments will be repeated changing the Adam optimizer for SGD. The experiments that will be repeated with this SGD are model 4 (with random flip) and model 6 (random flip and random rotate) without scheduler, with LR Step and with OneCycle - this makes a total of 6 new experiments. The first step of this new set of experiments is the model with the two transformations without a scheduler. The expected results might be similar to experiment 6 - a good validation accuracy, not the best, and a small overfitting.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp11PointNet.png "Accuracy and Loss")

ID  | Normalization | Dropout | Data Aug. | Optimizer | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: 
11 | Yes | Yes, p=0.3 | Random Fip+Rot. | SGD | 0.96115

**Findings**

Compared to experiment 6, which had the same configuration but using Adam optimizer instead, it can be seen that a better validation accuracy is obtained. However, the   overfitting has increased with respect to that experiment even though the same data augmentation techniques are being used.

#### Experiment 12: Random Flip + Random Rotation + LR Step + SGD Optimizer

**Setup and hypothesis**

A scheduler is applied to the previous model. The first used scheduler is the one referred in the paper, the LR Step. The data augmentation techniques are the same as the previous experiment, so having a better validation result is expected, but not an improvement of the overfitting. 

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp12PointNet.png "Accuracy and Loss")

ID  | Normalization | Dropout | Data Aug. | Optimizer | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:  
12 | Yes | Yes, p=0.3 | Random Fip+Rot. | SGD | LR Step | 0.96241

**Findings**

As expected, a better validation accuracy is obtained with respect to not using a scheduler. Although this result is still worse than expected, the best validation accuracy is very good. 

#### Experiment 13: Random Flip + Random Rotation + OneCycle + SGD Optimizer

**Setup and hypothesis**

Similar to the previous experiment, the same configuration as experiment 11 will be used but applying a scheduler, in this case the OneCycle. The configuration using random flip, random rotation and OneCycle scheduler was the best when using Adam optimizer. After seeing the previous results where SGD optimizer has improved the validation accuracy, obtain the best accuracy from this experiment is expected

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp13PointNet.png "Accuracy and Loss")

ID  | Normalization | Dropout | Data Aug. | Optimizer | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:  
13 | Yes | Yes, p=0.3 | Random Fip+Rot. | SGD | OneCycle | 0.96867

**Findings**

The hypothesis has failed, this experiment has obtained a very good accuracy validation but it is not the best so far. The overfitting is very small, but experiment 10 still has the best results.

#### Experiment 14: Random Flip + SGD Optimizer

**Setup and hypothesis**

The same configuration as experiment 4 is used, but in this case the used optimizer is SGD instead of Adam. The configuration using random flip was the best when using Adam optimizer and no learning scheduler. After seeing the previous results where SGD optimizer has improved the validation accuracy, obtaining the best accuracy from this experiment without using any learning scheduler is expected

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp14PointNet.png "Accuracy and Loss")

ID  | Normalization | Dropout | Data Aug. | Optimizer | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: 
14 | Yes | Yes, p=0.3 | Random Fip | SGD | 0.96115

**Findings**

The hypothesis has not been successful, the validation accuracy is worse than the one in experiment 4. Moreover, overfitting can be appreciated in the plots.
#### Experiment 15: Random Flip + LR Step + SGD Optimizer

**Setup and hypothesis**

The same configuration as experiment 14 is used, but in this case with a LR Step scheuler. Like in previous experiments, the learning rate will be divided by 2 every 20 epochs. The configuration using random flip was the best so far. After seeing the previous results where using a step learning rate has improved performance, having better results than in the previous experiment is expected to happen. 

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp15PointNet.png "Accuracy and Loss")

ID  | Normalization | Dropout | Data Aug. | Optimizer | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:  
15 | Yes | Yes, p=0.3 | Random Fip | SGD | LR Step | 0.95865       

**Findings**

The hypothesis has failed, a worse validation accuracy than in the previous experiment has been obtained. This validation accuracy is also worse than the one using the same configuration but using the Adam optimizer.

#### Experiment 16: Random Flip + OneCycle + SGD Optimizer

**Setup and hypothesis**

The same configuration as experiment 15 is used, but using OneCycle scheduler instead of  LR Step scheuler. Like in previous experiments, the learning rate will be divided by 2 every 20 epochs. The configuration using random flip and OneCycle scheduler was worse than the one using LR step instead but was better than the one without any scheduler. Therefore, an improvement of the results from experiment 14 is expected. 

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp16PointNet.png "Accuracy and Loss")

ID  | Normalization | Dropout | Data Aug. | Optimizer | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:  
16 | Yes | Yes, p=0.3 | Random Fip | SGD | One Cycle | 0.95238     

**Findings**

The hypothesis has failed, a worse validation accuracy than in the previous experiment is obtained. Consequently, it is also worse than in experiment 14, that was the original baseline. 

***

General conclusions and final choice of model is detailed and justified in the Conclusions section.

***

<a name="expgcn"></a>
### 4.3. Experiments with Graph Convolutional Network

The second part of the project consists of replacing PointNet to GCN. The first step is trying one layer of GCN and then further layers until three are added in order to compare results and extract the best model.  Once the baseline of the model is chosen, regularization techniques are added if needed, as well as schedulers and optimizer change, equivalent to PointNet procedure.

Labelled models follow the next notation:

* GCN1_ / GCN2_ / GC3_ refers to the baseline model, graph convolutional network with 1, 2 or 3 layers respectively
* D means applied Dropout
* F means applied Flip transform
* R means applied Rotation transform
* FR means both Flip and Rotation transforms applied
* S means scheduler. No subindex means LR Step. Sc refers to OneCycle scheduler
* O means optimizer. No O means Adam optimizer by default. Os refers to SGD optimizer

#### Experiment 1: GCN 1 layer

**Setup and hypothesis**

The first experiment of GCN consists of replacing PointNet GCN architecture with one GCN layer. In this experiment no dropout or any other regularization technique is being applied. There are no expected results from this experiment since it is the first one of the GCN series.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp1GCN.png "Accuracy and Loss")

Model  | Normalization | Best Val Accuracy 
:-------------: | :-------------:  | :-------------:
GCN1 | Yes | 0.8571 

**Findings**

It can be seen that the validation accuracy can still improve compared to PointNet results. By adding more layers, better performance will be expected.

#### Experiment 2: GCN 2 layers

**Setup and hypothesis**

Similar to the previous experiment, in this one GCN architecture is used, but with 2 layers instead of 1. Adding one more layer might increase the validation accuracy, starting to obtain results similar to PointNet.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp2GCN.png "Accuracy and Loss")

Model  | Normalization | Best Val Accuracy
:-------------: | :-------------:  | :-------------: 
GCN2 | Yes |  0.9185 | 0.858

**Findings**

The validation accuracy has improved with respect to experiment 1 due to the fact that the extra added 1 GCN layer, but some overfitting appeared. This is not a reason to stop adding one extra layer in the next experiment, since overfitting can be later adjusted with regularization techniques.

#### Experiment 3: GCN 3 layers

**Setup and hypothesis**

The final baseline experiment consists of adding the third GCN layer to the model and comparing this result with experiment 2, which had a better accuracy than experiment 1. Adding one GCN has improved the validation accuracy previously, so this experiment may be the one with best results.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp3GCN.png "Accuracy and Loss")

Model  | Normalization | Best Val Accuracy  
:-------------: | :-------------:  | :-------------: 
GCN3 | Yes | 0.9311 | 0.883

**Findings**

It can be observed from the results that this third experiment has the best validation accuracy performance although it shows an overfitting problem. As mentioned before, this model will be the chosen baseline and from now on the objective will be trying to fix overfitting without reducing the accuracy. 

#### Experiment 4: Dropout

**Setup and hypothesis**

In order to reduce the overfitting shown by the last experiment, the first applied regularization technique will be dropout, according to the same procedure as with PointNet.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp4GCN.png "Accuracy and Loss")

Model | Normalization | Dropout | Best Val Accuracy
:-------------: | :-------------:  | :-------------: | :-------------:
GCN3_D | Yes | Yes, p=0.3 | 0.9348 | 0.8799

**Findings**

As it can be seen in the plots, dropout seems not to solve the overfitting problem. Nonetheless, it will be kept because the validation accuracy has improved. Furthermore, another regularization method will be applied: data augmentation. As it was done in PointNet, random flip and random rotate will be applied separately and jointly.

#### Experiment 5: Random Flip

**Setup and hypothesis**

The data augmentation method is used to reduce overfitting. In this experiment, only random flip is applied. It is expected to reduce overfitting while keeping the same validation accuracy.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp5GCN.png "Accuracy and Loss")

Model | Normalization | Dropout | Data Aug. | Best Val Accuracy
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------:
GCN3_DF | Yes | Yes, p=0.3 | Random Fip | 0.9487

**Findings**

Random flip seems to reduce overfitting with respect to only using dropout as well as it increases the validation accuracy. Next step will be to keep using data augmentation in order to see what model gets the best results.

#### Experiment 6: Random Rotation

**Setup and hypothesis**

This experiment uses the same configuration as experiment 5, but we will change the random flip transformation to random rotation. As seen in previous results, it is expected from random rotation to also help reducing overfitting while keeping the same validation accuracy.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp6GCN.png "Accuracy and Loss")

Model | Normalization | Dropout | Data Aug. | Best Val Accuracy
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: 
GCN3_DR | Yes | Yes, p=0.3 | Random Rot. | 0.9147    

**Findings**

Random rotation reduces the overfitting, but the result of validation accuracy is worse than in previous experiment, so this transformation doesn’t help as it does not help to get the best result. 

#### Experiment 7: Random Flip and Random Rotation

**Setup and experiments**

After analysing the results from random flip and random rotation, the same configuration as in previous experiments (3 layers of GCN and dropout) is applied but the two transformations at the same time. As before, reducing the overfitting without a decrease of validation accuracy is the expected result.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp7GCN.png "Accuracy and Loss")

Model | Normalization | Dropout | Data Aug. | Best Val Accuracy
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: 
GCN3_DFR | Yes | Yes, p=0.3 | Random Flip+Rot. | 0.9023

**Findings**

The results show that both transformations seem to perform similarly to by just applying rotation. Nonetheless, the validation accuracy has decreased. The experiment 5 with just random flip transformation has the best result, but models from experiment 5 and 7 will be kept as it was done in the PointNet experiments.

#### Optimizer and learning rate

Similar as in the PointNet experiments, the next objective will be looking for the best configuration of the optimizer and learning rate using two different schedulers: the one used in the paper (LR Step) and another one to compare results (OneCycle). In these next experiments the setup of experiments 5 and experiments 7 will be used, the models with the same configurations that were used in PointNet. Applying the schedulers on both models will allow to see if a model with the highest validation accuracy without having overfitting can be achieved.

#### Experiment 8: Random Flip + LR Step 

**Setup and hypothesis**

A scheduler is starting to be used at the training loop. In this experiment the scheduler used is LR step, which will decrease by 2 the learning rate every 20 epochs. The model used is the one from experiment 5 which has the random flip transformation. This is the model with higher validation accuracy. The expected result from this experiment is increasing the validation accuracy as the scheduler is not used to reduce overfitting.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp8GCN.png "Accuracy and Loss")

Model | Normalization | Dropout | Data Aug. | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------:
GCN3_DFS | Yes | Yes, p=0.3 | Random Flip | LR Step | 0.9223

**Findings**

From the results, it can be appreciated that the scheduler doesn't help to improve the validation accuracy but it is useful to smooth the oscillations during training.

#### Experiment 9: Random Flip + Random Rotation + LR Step

**Setup and hypothesis**

As explained before, this experiment will be the one that applies both transformations and the scheduler. Having a model with a higher validation accuracy than the previous experiment is expected to happen.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp9GCN.png "Accuracy and Loss")

Model | Normalization | Dropout | Data Aug. | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------:
GCN3_DFRS | Yes | Yes, p=0.3 | Random Flip+Rot | LR Step | 0.8922

**Findings**

The validation accuracy of this experiment is lower with respect to the previous one (applying only flip transformation) and it is also lower than the same model without the scheduler. It can be concluded that the LR Step scheduler might not be the best one to improve accuracy in this scenario. 

#### Experiment 10: Random Flip + OneCycle

**Setup and hypothesis**

Now that the performance of the scheduler used in the reference paper has been tested, a different one will be applied as it was done in PoitNet experiments. In the following experiments the scheduler used is OneCycle. This experiment is used with the configuration of experiment 5 (using random flip transformation) and it is expected to give better results than in experiment 9, where LR Step was tested.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp10GCN.png "Accuracy and Loss")

Model | Normalization | Dropout | Data Aug. | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------:
GCN3_DFSc | Yes | Yes, p=0.3 | Random Flip | OneCycle | 0.9360

**Findings**

With the OneCycle scheduler, oscillations almost disappear when compared with other experiments. Thanks to the data augmentation, the level of overfitting seems to also be reduced. This is the best result so far in terms of both validation accuracy and overfitting avoidance combined. It still doesn’t reach the best validation accuracy of the model 5.

#### Experiment 11: Random Flip + Random Rotation + OneCycle

**Setup and hypothesis**

Like it was done with the LR Step scheduler, OneCycle is tested with both transforms of data augmentation. The expected results are similar to the previous experiment, but it might show a lower validation accuracy.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp11GCN.png "Accuracy and Loss")

Model | Normalization | Dropout | Data Aug. | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------:
GCN3_DFRSc | Yes | Yes, p=0.3 | Rnd Flp+Rot | OneCycle | 0.8860

**Findings**

Similarly with experiment 11, this scenario reduces overfitting because of the data augmentation. However, the validation accuracy is worse than the previous experiment where only the flip transformation was applied.

#### Experiment 12: Random Flip + SGD

**Setup and hypothesis**

Similarly with the PointNet procedure and as mentioned before, some experiments will be repeated but changing the Adam optimizer for SGD in order to find the optimum one for this scenario. The experiments that will be repeated with this SGD are number 5 (with RandomFlip) and number 7 (random flip and random rotate) without scheduler, with LR Step and with OneCycle - this sums up a total of 6 new experiments. 

The first one will be the model with only random flip transformation without any scheduler. The expected results might be similar to experiment 5, i.e.,  a good validation accuracy and a small overfitting.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp12GCN.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Optimizer | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------:
GCN3_DFOs | Yes | Yes, p=0.3 | Random Flip | SGD | 0.9160  

**Findings**

From the results it can be seen a worse performance in terms of validation accuracy than in experiment 5 (where random flip and Adam optimizer were used). Nonetheless, there is an improvement regarding overfitting level in this case.

#### Experiment 13: Random Flip + Random Rotation + SGD

**Setup and hypothesis**

The same configuration as experiment 7 is used, but in this case SGD optimizer is used instead of Adam.  The configuration using random flip was the best when using Adam optimizer and no learning scheduler. After seeing the previous results where SGD optimizer has decreased the validation accuracy, it is expected from this experiment to obtain a model with lower validation accuracy than before, but reducing overfitting.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp13GCN.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Optimizer | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------:
GCN3_DFROs | Yes | Yes, p=0.3 | Rnd Flp+Rot | SGD | 0.8822

**Findings**

The performance of this experiment is worse in terms of validation accuracy than in experiment 7 (where random flip and random rotation with Adam optimizer were used). Also, it can not be appreciated any improvement regarding overfitting level in this case.

#### Experiment 14: Random Flip + LR Step + SGD

**Setup and hypothesis**

The first step will be applying a scheduler to the previous model. The chosen one is the one used in the reference paper, LR Step. The data augmentation technique applied in this case is only random flip, so it is expected to have a better validation result, but not an improvement of the overfitting. 

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp14GCN.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Optimizer | Scheduler | Best Val Accuracy
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:
GCN3_DFSOs | Yes | Yes, p=0.3 | Random Flip | SGD | LR Step | 0.9223

**Findings**

The exact same validation accuracy result than in experiment 8 is obtained, where the same configuration (random flip and LR scheduler) was used, but with Adam optimizer instead. Nonetheless, in this case there is more level of overfitting, so SGD does not perform better in this scenario.

#### Experiment 15: Random Flip + Random Rotation + LR Step + SGD

**Setup and hypothesis**

In this experiment, the test of SDG with data augmentation and LR Step scheduler will continue. Now is the turn of joining random flip and random rotation transformations. It is expected to have a model with a higher validation accuracy than in the previous experiment.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp15GCN.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Optimizer | Scheduler | Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:
GCN3·_DFRSOs | Yes | Yes, p=0.3 | Random Flip+Rot. | SGD | LR Step | 0.8609

**Findings**

As it can be seen in the results table, the validation accuracy is lower than experiment 9, where the model had the same configuration but using Adam optimizer instead. The overfitting level has improved in this experiment, due to the fact that the two data augmentation transformations and the SGD optimizer are being used.

#### Experiment 16: Random Flip + OneCycle + SGD

**Setup and hypothesis**

Continuing with applying schedulers in the network, OneCycle will be used instead of LR Step. The configuration using random flip, random transformation and OneCycle scheduler with Adam optimizer was not a good solution to increase the validation accuracy. After analysing those results it is expected to obtain a worse validation accuracy than in experiment 14, where LR Step was used.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp16GCN.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Optimizer | Scheduler |Best Val Accuracy  
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:
GCN3·_DFScOs | Yes | Yes, p=0.3 | Random Flip. | SGD | OneCycle | 0.9223 

**Findings**

Applying OneCycle scheduler, oscillations almost disappear compared to other experiments. The level of overfitting seems to also be reduced thanks to applying random flip transformation. However, the validation accuracy result is not better when compared with Adam optimizer.

#### Experiment 17: Random Flip + Random Rotation + OneCycle + SGD

**Setup and hypothesis**

Like it was done with the LR Step scheduler, OneCycle is tested with the model where random flip and random rotation transforms are applied. The expected results are similar to the previous experiment, but there might be a lower validation accuracy.

**Results**

![](https://github.com/g4aidl-upc-spring-2021/3D-PointCloud-Classification/blob/master/images/Exp17GCN.png "Accuracy and Loss")

ID | Normalization | Dropout | Data Aug. | Optimizer | Scheduler |Best Val Accuracy 
:-------------: | :-------------:  | :-------------: | :-------------: | :-------------: | :-------------: | :-------------:
GCN·_DFRScOs | Yes | Yes, p=0.3 | Random Flip+Rot.. | SGD | OneCycle | 0.8672 

**Findings**

This last experiment reduces overfitting when compared with previous experiments, maybe because of the use of the two data augmentation techniques. However, the validation accuracy is lower than the accuracy of the model when using Adam optimizer. 

***

General conclusions and final choice of model is detailed and justified in the Conclusions section.

***

<a name="conc"></a>
## 5. Conclusions

<a name="expconc"></a>
### 5.1. Experiments Conclusions

After analysing all experiments from PointNet and GCN, we will proceed to compare the best models of each architecture based on the validation accuracy. 

#### Best model from PointNet

The highest accuracy from all the PointNet experiments is __0.96992__ and it corresponds to __experiment 10__, which has the following configuration:

* Normalization
* Adam optimizer with weight decay = 1e-3
* Dropout of 0.3
* Random Flip + Random Rotation
* OneCycle Scheduler

#### Best model from Graph Convolutional Network

For GCN. we chose the experiments with validation accuracy of __0.9360__ which corresponds to __experiment 10__ and has the following configuration:

* Normalization
* Adam optimizer with weight decay = 1e-3
* Dropout of 0.3
* Random Flip
* OneCycle Scheduler

This is not the best model in terms of highest validation accuracy, but it is the one which has a better balance between this metric and overfitting reduction. The best model in terms of validation accuracy is experiment 5, with a value of 0.9487. We consider it is worth this difference of 0.01 in accuracy having this much improvement of overfitting reduction.

#### Test results from both architectures

Now it’s time to use these models with test split and analyse the behaviour of the models with unseen data.

Model | Test accuracy
:-------------: | :-------------: 
PointNet | 0.9240 
GCN | 0.8766

Analysing the test accuracy we can conclude that the __PointNet architecture is better than GCN when performing the classification task of ModelNet__. 

On one hand, this can be due to the simple fact that PointNet is a more complex architecture, specially designed for this task in which several steps have been considered in order to both achieve the final objective and solve the constraints that the problem presents.

Furthermore, if we take into consideration the number of trainable weights of each architecture:

Trainable Weights for PointNet | Trainable Weights of Graph Convolutional Network
:-------------: | :-------------: 
3.465.305 | 3.594

We can see that PointNet network has much more capacity to learn abstract data representation rather than Graph Convolutional Network. However, it is surprisingly interesting to note that PointNet has an improvement of only 5% in accuracy performance with an amount of two orders of magnitude more parameters than GCN.  

<a name="proconc"></a>
### 5.2. Project Conclusions

* In general, good practices require time.
	- From data understanding to fine-tuning process, time becomes tight when developing an end-to-end solution in some weeks.
* Even though they might appear to be a hard-to-handle structure, __point clouds and meshes__ turned out to be an easier type of data to work with than expected, despite our early thoughts.
	- Point clouds are simply defined by a tensor of coordinates, although one might take care when dealing with big batches of data.
	- Graphs are defined by both coordinates and edges. Although adjacent matrix of edges might not be easy to understand mathematically, it is easy to work with them with the functions implemented in Pytorch Geometric.
	- In both cases, data preprocessing is an essential step of the solution. We noted how by simply adding a normalization of the input data, output improved drastically.
* Implementing and training a complex architecture from scratch is not a straightforward task even if you have a guide such as a paper publication.
	- Errors may arise when not considering subtle details about data treatment inside the network.
	- It requires time and resources. Sometimes it is better to start with a simpler design or dataset.
* PointNet outperformed Graph Convolutional Networks in classification task:
	- We got reasonable good results from each of them through the fine-tuning process.
	- We noted the importance of choosing different features for the training such as optimizers (in which Adam turned out to be the best) in order to reach the best performance.

<a name="milestones"></a>
## 6. Achieved Milestones

Through the implementation of these networks and the execution of all the experiments that are documented, we have been able to accomplish the objectives that we proposed and even some extra:

* We have learnt to work with 3D point clouds but also with graphs. We have been able to process the data, use it in the networks and visualize it.
* We have reproduced the PointNet architecture of a scientific publication from scratch. Although we firstly implemented it for segmentation task, we have been able to manage 
the redesign also for classification.
* Besides PointNet, we have developed and implemented a classifier network architecture based on Graph Convolutional Networks.
* We have been able to train these two classifiers (PointNet and GCN) in order to recognise different classes of objects from ModelNet by executing a total of 33 experiments.
* All the results from these experiments have been analysed using validation accuracy and have been improved depending on their performance and results. Drawing conclusions and improving the experiments has led us to find the best model.

<a name="fwork"></a>
## 7. Further Work

This project has been focused on reproducing the PointNet classifier from scratch and then implementing a GCN network also for classification purposes, but it could be extended in some different ways:

* Instead of ModelNet10, use ModelNet40 which contains meshes for 40 types of different objects or other datasets of point clouds such as [ShapeNet](https://shapenet.org/).
* Equal number of elements of each class in the dataset in order to have a balance amount of objects.
* Try different values for k in KNNGraph algorithm to generate graphs from point clouds and try to generate graph from radius.
* Implement other metrics for evaluating the performance of the models, such as precision and recall, classification error or F1 score.
* Try other regularization techniques, such as different transformations in data augmentation such as [Random Scale](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RandomScale).
* Try different values for the hyper parameters of the training process such as learning rate, batch size or number of workers.
* Try different values for the parameters of the implemented optimizers and schedulers - for example, weight decay for Adam optimizer, momentum for SGD optimizer, multiplicative factor for schedulers, and so on.
* Implement deeper GCN network.
* Implement new architecutres that use graphs, such as Graph Attention Networks (GAN's, do not confuse with Generative Adversarial Network).
* We could even extend the scope of the project by attempting a segmentation task instead of doing classification. We could restore the code and add it to the PointNet class, adapting the network to use another dataset.

<a name="ref"></a>
## 8. References

All references regarding papers consulted and libraries used are directly linked to beforehand mentions.

The main two papers from Neural Networks architecutres are:

__PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation | Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas__

__Semi-Supervised Classification with Graph Convolutional Networks | Thomas N. Kipf, Max Welling__

