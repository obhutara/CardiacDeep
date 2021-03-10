# 4D Flow Cardiovascular Segmentation
3D U-Net for multi-class cardiac segmentation and Bayesian 3D U-Net for uncertainty estimation

4D MRI's 220 patients' thorax was made available at CMIV (Center for Medical Image science and Visualisation)

## Multi-Class Segmentation using 3D U-Net

- Use the code in 3dunet.py which reads in the data, sets up required libraries from torchio and monai for the unet architecture, data preprocessing, data augmentation and training of the neural network.
- Outputs the results in Nifti files for visualisation and plotting of Dice scores by each class. Also see infy.py
- transforms.py is for visualisation of different data augmentation techniques to experiment with to create the an optimal chain of augmentation techniques to be applied.

## Uncertainty estimation using Bayesian 3D U-Net

- Use the files within the folder Bayesian 3D U-Net
- Set up the required version of tensorboard libraries using tensorboardcode.py and selfcheck.py, Set up using configs.json, Generate data using generate_data.py, run the train_test.sh shell script to run all the code.
