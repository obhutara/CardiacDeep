
#Tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision


# constant for classes
classes = ('Left Ventricle', 'Right Ventricle', 'Left Atrium', 'Pulmonary artery ', 'Aorta', 'Right Atrium')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


writer = SummaryWriter('runs/3dunet_experiment_1')

images, labels = prepare_batch(batch,training_batch_size, device)

# create grid of images
img_grid = torchvision.utils.make_grid(images[0])

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('3dunet_images', img_grid)

#now running ; tensorboard --logdir=runs



torchvision.utils.save_image(tensor=images[0],fp=images_dir)