# ML_ND3_CapstoneProject
Capstone project of the Udacity ML Nanodegree

Mitosis Detection using Transfer Learning in PyTorch

The capstone project contains a basic solution for the Mitosis Detection problem encountered in Digital Pathology. It falls into the wider field of AI in Healthcare.<br/>
The inspiration for this project came from the Tumor Proliferation Assesment Challenge (TUPAC) 2016. Also credit is given to participant teams in the competition. Their papers are referenced in the project report. 

The dataset used as initial input is provided by Mr. Andrew Janowczyk on his personal blog and can be downloaded from the following location:
http://www.andrewjanowczyk.com/deep-learning/  - Mitosis Detection.<br/>
The instructions provided on his blog constituted a great source of help in developing a course of action for this project, as well as providing valuable context information and implementation details.

The libraries used in the project are: 
- Pytorch - torchvision
- OpenCv - cv2
- Numpy
- Scikit learn - metrics

The project is structured into **two** Jupiter Notebooks:
1. *MakePatches*
2. *Train_Model*

An additional file containing helper functions is included: *helper_functions.py*

Note: The selection of samples to form the train, validation and test datasets is done manually. Thus, the folders that will contain the datasets are not created programatically.
