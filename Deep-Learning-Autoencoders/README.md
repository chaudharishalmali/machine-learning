# machine-learning
This project was for the study of auto-encoders and deep neural networks by creating a face recognition system. In this project, I first trained auto-encoders to extract the features from the images. For this, I tried several configurations of auto-encoders with different number of nodes and also by stacking 2 and 3 auto-encoders. After extracting the features, I used these features to train network consisting of stacked auto-encoders with best configurations. When the training was done, I added Soft-max layer to form a deep neural network. I fine tuned this network with back propagation and did the testing using the test images. 
Database used: AT&T face image database - 40 subjects. 10 Images of each subject. Total 400 images 
Scripting: MATLAB with neural network and digital image processing toolboxes 
The project was done in two modes: 
Mode 1: Trained one deep neural network for whole dataset 
Mode 2: Trained a deep neural network for for each subject separately 
For both modes first six images of all subjects were used fr training and remaining four images of all subjects were used for testing.
