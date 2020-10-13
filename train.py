#pip install torch highresnet
#pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#pip install --quiet --upgrade pip
#pip install --quiet --upgrade niwidgets
#pip install --quiet --upgrade highresnet
#pip install --quiet --upgrade unet
#pip install --quiet --upgrade torchio
#pip install -qq tree


seed = 42
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
num_epochs = 5

# If the following values are False, the models will be downloaded and not computed
compute_histograms = True
train_whole_images = True
train_patches = True

import copy
import enum
import random
random.seed(seed)
import warnings
import tempfile
import subprocess
import multiprocessing
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
torch.manual_seed(seed)

import torchio as tio
from torchio import AFFINE, DATA

import numpy as np
import nibabel as nib
from unet import UNet
from scipy import stats
import SimpleITK as sitk
import matplotlib.pyplot as plt

from IPython import display
from tqdm.notebook import tqdm
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)

print('TorchIO version:', tio.__version__)


#@title (Helper functions, double-click here to expand)
def show_nifti(image_path_or_image, colormap='gray'):
    try:
        from niwidgets import NiftiWidget
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            widget = NiftiWidget(image_path_or_image)
            widget.nifti_plotter(colormap=colormap)
    except Exception:
        if isinstance(image_path_or_image, nib.AnalyzeImage):
            nii = image_path_or_image
        else:
            image_path = image_path_or_image
            nii = nib.load(str(image_path))
        k = int(nii.shape[-1] / 2)
        f = plt.figure()
        plt.imshow(nii.dataobj[..., k], cmap=colormap)
        f.show()
        #plt.imshow(nii.dataobj[..., k], cmap=colormap) (for jupyter note)

def show_subject(subject, image_name, label_name=None):
    if label_name is not None:
        subject = copy.deepcopy(subject)
        affine = subject[label_name].affine
        label_image = subject[label_name].as_sitk()
        label_image = sitk.Cast(label_image, sitk.sitkUInt8)
        border = sitk.BinaryContour(label_image)
        border_array, _ = tio.utils.sitk_to_nib(border)
        border_tensor = torch.from_numpy(border_array)[0]
        image_tensor = subject[image_name].data[0]
        image_tensor[border_tensor > 0.5] = image_tensor.max()
    with tempfile.NamedTemporaryFile(suffix='.nii',delete=False) as f:
        subject[image_name].save(f.name)
        show_nifti(f.name)

def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
    values = tensor.numpy().ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)



# Dataset
dataset_path = 'E:\\segment_data\\3D_data'
dataset_dir = Path(dataset_path)
histogram_landmarks_path = 'landmarks.npy'
print(dataset_dir)

#Datasets nifti stuff,
images_dir = dataset_dir /'magnitude_3D'
labels_dir = dataset_dir /'segmentation_3D'
image_paths = sorted(images_dir.glob('*.nii.gz'))
label_paths = sorted(labels_dir.glob('*.nii.gz'))
assert len(image_paths) == len(label_paths)


subjects = []
for (image_path, label_path) in zip(image_paths, label_paths):
    subject = tio.Subject(
        mri=tio.ScalarImage(image_path),
        heart=tio.LabelMap(label_path),
    )
    subjects.append(subject)
dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')


#Having a look at a sample
one_subject = dataset[0]
print(one_subject)
print(one_subject.mri)

#Normalisation
landmarks = tio.HistogramStandardization.train(
    image_paths,
    output_path=histogram_landmarks_path,
)
np.set_printoptions(suppress=True, precision=3)
print('\nTrained landmarks:', landmarks)

#Histogram standardisation
#Hist standardization
landmarks_dict = {'mri': landmarks}
histogram_transform = tio.HistogramStandardization(landmarks_dict)

#Z-Norm
znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)

sample = dataset[0]
transform = tio.Compose([histogram_transform, znorm_transform])
znormed = transform(sample)

fig, ax = plt.subplots(dpi=100)
plot_histogram(ax, znormed.mri.data, label='Z-normed', alpha=1)
ax.set_title('Intensity values of one sample after z-normalization')
ax.set_xlabel('Intensity')
ax.grid()


training_transform = Compose([
  #  ToCanonical(),
  #  Resample(4),
    CropOrPad((112, 112, 48), padding_mode=0), #reflect
  #  RandomMotion(),
  #  HistogramStandardization({'mri': landmarks}),
  #  RandomBiasField(),
  #  ZNormalization(masking_method=ZNormalization.mean),
  #  RandomNoise(),
  #  RandomFlip(axes=(0,)),
  #  OneOf({
  #      RandomAffine(): 0.8,
  #      RandomElasticDeformation(): 0.2,
  #  }),
])

validation_transform = Compose([
  #  ToCanonical(),
  #  Resample(4),
    CropOrPad((112, 112, 48), padding_mode=0),
  #  HistogramStandardization({'mri': landmarks}),
  #  ZNormalization(masking_method=ZNormalization.mean),
])

num_subjects = len(dataset)
num_training_subjects = int(training_split_ratio * num_subjects)

training_subjects = subjects[:num_training_subjects]
validation_subjects = subjects[num_training_subjects:]

training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')



#@title (Deep learning functions, double-click here to expand)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def prepare_batch(batch, training_batch_size, device):
    inputs = batch['mri'][DATA].to(device)
    foreground = batch['heart'][DATA].to(device)
    foreground1 = torch.where(foreground == 1, torch.ones(training_batch_size, 1, 112, 112, 48).to(device), torch.zeros(training_batch_size, 1, 112, 112, 48).to(device)).to(device)
    foreground2 = torch.where(foreground == 2, torch.ones(training_batch_size, 1, 112, 112, 48).to(device), torch.zeros(training_batch_size, 1, 112, 112, 48).to(device)).to(device)
    foreground3 = torch.where(foreground == 3, torch.ones(training_batch_size, 1, 112, 112, 48).to(device), torch.zeros(training_batch_size, 1, 112, 112, 48).to(device)).to(device)
    foreground4 = torch.where(foreground == 4, torch.ones(training_batch_size, 1, 112, 112, 48).to(device), torch.zeros(training_batch_size, 1, 112, 112, 48).to(device)).to(device)
    foreground5 = torch.where(foreground == 5, torch.ones(training_batch_size, 1, 112, 112, 48).to(device), torch.zeros(training_batch_size, 1, 112, 112, 48).to(device)).to(device)
    foreground6 = torch.where(foreground == 6, torch.ones(training_batch_size, 1, 112, 112, 48).to(device), torch.zeros(training_batch_size, 1, 112, 112, 48).to(device)).to(device)
    background = 1 - (foreground1+foreground2+foreground3+foreground4+foreground5+foreground6)
    targets = torch.cat((background,foreground1,foreground2,foreground3,foreground4,foreground5,foreground6), dim=CHANNELS_DIMENSION)
    return inputs, targets

def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score

def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)

def forward(model, inputs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        logits = model(inputs)
    return logits

def get_model_and_optimizer(device):
    model = UNet(
        in_channels=1,
        out_classes=7,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer

def run_epoch(epoch_idx, action, loader, model, optimizer):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    for batch_idx, batch in enumerate(tqdm(loader)):
        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = forward(model, inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = get_dice_loss(probabilities, targets)
            batch_loss = batch_losses.mean()
            if is_training:
                batch_loss.backward()
                optimizer.step()
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')

def train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem):
    run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer)
    for epoch_idx in range(1, num_epochs + 1):
        print('Starting epoch', epoch_idx)
        run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer)
        run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer)
        torch.save(model.state_dict(), f'{weights_stem}_epoch_{epoch_idx}.pth')



training_instance = training_set[42]  # transform is applied in SubjectsDataset
show_subject(training_instance, 'mri', label_name='heart')

validation_instance = validation_set[42]
show_subject(validation_instance, 'mri', label_name='heart')

training_batch_size = 8
validation_batch_size = 2 * training_batch_size

training_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=training_batch_size,
    shuffle=True,
    num_workers=multiprocessing.cpu_count(),
)

validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=validation_batch_size,
    num_workers=multiprocessing.cpu_count(),
)


one_batch = next(iter(training_loader))

k = 12
batch_mri = one_batch['mri'][DATA][..., k]
batch_label = one_batch['heart'][DATA][..., k]
slices = torch.cat((batch_mri, batch_label))
image_path = 'batch_whole_images.png'
save_image(slices, image_path, nrow=training_batch_size//2, normalize=True, scale_each=True, padding=0)
display.Image(image_path)





#TRAIN

model, optimizer = get_model_and_optimizer(device)

if train_whole_images:
    weights_stem = 'whole_images'
    train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem)
#else:
 #  weights_path = 'C:\\users\\omkbh\\Downloads\\whole_images_epoch_5.pth'
 #  #weights_url = 'https://www.dropbox.com/s/h0yxbzfncjj84ep/whole_images_epoch_5.pth?dl=0'
 #  #!curl --location --silent --output {weights_path} {weights_url}
 #  model.load_state_dict(torch.load(weights_path))


#Outputs the image for ifnerence nifti slice
batch = next(iter(validation_loader))
model.eval()
inputs, targets = prepare_batch(batch,training_batch_size, device)
with torch.no_grad():
    logits = forward(model, inputs)
labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
k = 12
batch_mri = inputs[..., k]
batch_label = labels[..., k]
slices = torch.cat((batch_mri, batch_label))
inf_path = 'inference.png'
save_image(slices, inf_path, nrow=training_batch_size//2, normalize=True, scale_each=True, padding=0)
display.Image(inf_path)