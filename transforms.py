#!pip install torchio==0.17.44 --quiet
#!pip install pandas --quiet
#!pip install matplotlib --quiet
#!pip install seaborn --quiet
#!pip install scikit-image --quiet
#!pip install Pillow --quiet
#!curl -s -o colormap.txt https://raw.githubusercontent.com/thenineteen/Semiology-Visualisation-Tool/master/slicer/Resources/Color/BrainAnatomyLabelsV3_0.txt
#!curl -s -o slice_7t.jpg https://static.healthcare.siemens.com/siemens_hwem-hwem_ssxa_websites-context-root/wcm/idc/groups/public/@global/@imaging/@mri/documents/image/mda5/nje2/~edisp/siemens-healthineers_mri_magnetom-terra_3-07093947/~renditions/siemens-healthineers_mri_magnetom-terra_3-07093947~8.jpg
#!curl -s -o slice_histo.jpg https://bcf.technion.ac.il/wp-content/uploads/2018/05/Neta-histology-slice-626.jpg
#!curl -s -o vhp.zip ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Visible-Human/Sample-Data/Six%20slices%20from%20the%20Visible%20Male.zip
#!unzip -o vhp.zip > /dev/null

import copy

import torch
import torchio as tio

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
%config InlineBackend.figure_format = 'retina'


# @title Visualization functions (double-click to expand)
def get_bounds(self):
    """Get image bounds in mm.

    Returns:
        np.ndarray: [description]
    """
    first_index = 3 * (-0.5,)
    last_index = np.array(self.spatial_shape) - 0.5
    first_point = nib.affines.apply_affine(self.affine, first_index)
    last_point = nib.affines.apply_affine(self.affine, last_index)
    array = np.array((first_point, last_point))
    bounds_x, bounds_y, bounds_z = array.T.tolist()
    return bounds_x, bounds_y, bounds_z


def to_pil(image):
    from PIL import Image
    from IPython.display import display
    data = image.numpy().squeeze().T
    data = data.astype(np.uint8)
    image = Image.fromarray(data)
    w, h = image.size
    display(image)
    image.show()
    print()  # in case multiple images are being displayed


def stretch(img):
    p1, p99 = np.percentile(img, (1, 99))
    from skimage import exposure
    img_rescale = exposure.rescale_intensity(img, in_range=(p1, p99))
    return img_rescale


def show_fpg(
        subject,
        to_ras=False,
        stretch_slices=True,
        indices=None,
        intensity_name='mri',
        parcellation=True,
):
    subject = tio.ToCanonical()(subject) if to_ras else subject

    def flip(x):
        return np.rot90(x)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    if indices is None:
        half_shape = torch.Tensor(subject.spatial_shape) // 2
        i, j, k = half_shape.long()
        i -= 55 # use a better slice
    else:
        i, j, k = indices
    bounds_x, bounds_y, bounds_z = get_bounds(subject.mri)  ###

    orientation = ''.join(subject.mri.orientation)
    if orientation != 'RAS':
        import warnings
        warnings.warn(f'Image orientation should be RAS+, not {orientation}+')

    kwargs = dict(cmap='gray', interpolation='none')
    data = subject[intensity_name].data
    slices = data[0, i], data[0, :, j], data[0, ..., k]
    if stretch_slices:
        slices = [stretch(s.numpy()) for s in slices]
    sag, cor, axi = slices

    axes[0, 0].imshow(flip(sag), extent=bounds_y + bounds_z, **kwargs)
    axes[0, 1].imshow(flip(cor), extent=bounds_x + bounds_z, **kwargs)
    axes[0, 2].imshow(flip(axi), extent=bounds_x + bounds_y, **kwargs)

    kwargs = dict(interpolation='none')
    data = subject.heart.data
    slices = data[0, i], data[0, :, j], data[0, ..., k]
    if parcellation:
        sag, cor, axi = [color_table.colorize(s.long()) if s.max() > 1 else s for s in slices]
    else:
        sag, cor, axi = slices
    axes[1, 0].imshow(flip(sag), extent=bounds_y + bounds_z, **kwargs)
    axes[1, 1].imshow(flip(cor), extent=bounds_x + bounds_z, **kwargs)
    axes[1, 2].imshow(flip(axi), extent=bounds_x + bounds_y, **kwargs)

    plt.tight_layout()


class ColorTable:
    def __init__(self, colors_path):
        self.df = self.read_color_table(colors_path)

    @staticmethod
    def read_color_table(colors_path):
        df = pd.read_csv(
            colors_path,
            sep=' ',
            header=None,
            names=['Label', 'Name', 'R', 'G', 'B', 'A'],
            index_col='Label',
        )
        return df

    def get_color(self, label: int):
        """
        There must be nicer ways of doing this
        """
        try:
            rgb = (
                self.df.loc[label].R,
                self.df.loc[label].G,
                self.df.loc[label].B,
            )
        except KeyError:
            rgb = 0, 0, 0
        return rgb

    def colorize(self, label_map: np.ndarray) -> np.ndarray:
        rgb = np.stack(3 * [label_map], axis=-1)
        for label in np.unique(label_map):
            mask = label_map == label
            color = self.get_color(label)
            rgb[mask] = color
        return rgb

color_table = ColorTable('colormap.txt')


fpg = one_subject #tio.dataset.FPG()
print('Sample subject:', fpg)
show_fpg(fpg)

to_ras = tio.ToCanonical()
fpg_ras = to_ras(fpg)
print('Old orientation:', fpg.mri.orientation)
print('New orientation:', fpg_ras.mri.orientation)
show_fpg(fpg_ras)

print(fpg_ras.mri)
print(fpg_ras.heart)

#Another handy use for the Resample transform is to apply a precomputed transformation to a standard space,
#such as the MNI space. The FPG dataset includes this transform:

np.set_printoptions(precision=2, suppress=True)
fpg.mri.affine #.numpy()

# To maybe CropOrPad
fpg_ras.mri.spatial_shape

target_shape = 96, 96, 48
crop_pad = tio.CropOrPad(target_shape)
show_fpg(crop_pad(fpg_ras))

#Random affine :To simulate different positions and size of the patient within the scanner, we can use a RandomAffine transform.
#To improve visualization, we will use a 2D image and add a grid to it.


image = tio.ScalarImage('slice_7t.png')
spacing = image.spacing[0]
image = tio.Resample(spacing)(image)
print('Downloaded slice:', image)
to_pil(image)



tio.ScalarImage('slice_7t.png').spacing

#grid over the image
slice_grid = copy.deepcopy(image)
data = slice_grid.data
white = data.max()
N = 16
data[..., ::N, :, :] = white
data[..., :, ::N, :] = white
to_pil(slice_grid)

#random affine (rotates the image with grid slightly anticlock)
random_affine = tio.RandomAffine(seed=43)
slice_affine = random_affine(slice_grid)
to_pil(slice_affine)

#random affine with zoom focusing on the heart
random_affine_zoom = tio.RandomAffine(scales=(1.3, 1.3), seed=43)
slice_affine = random_affine_zoom(slice_grid)
to_pil(slice_affine)

#Random flip
#Flipping images is a very cheap way to perform data augmentation.
#In medical images, it's very common to flip the images horizontally.
#We can specify the dimensions indices when instantiating a RandomFlip transform.
#However,if we don't know the image orientation, we can't know which dimension
#corresponds to the lateral axis. In TorchIO, you can use anatomical
#labels instead, so that you don't need to figure out image orientation
#to know which axis you would like to flip. To make sure the transform modifies
#the image, we will use the inferior-superior (longitudinal) axis and a flip
#probability of 1. If the flipping happened along any other axis, we might not
#notice it using this visualization.

random_flip = tio.RandomFlip(axes=['inferior-superior'], flip_probability=1)
fpg_flipped = random_flip(fpg_ras)
show_fpg(fpg_flipped)

#Random elastic deformation To simulate anatomical variations in our images,
#we can apply a non-linear deformation using RandomElasticDeformation.

max_displacement = 10, 10, 0  # in x, y and z directions
random_elastic = tio.RandomElasticDeformation(max_displacement=max_displacement, seed=0)
slice_elastic = random_elastic(slice_grid)
to_pil(slice_elastic)

#As explained in the documentation, one can change the number of grid control
#points to set the deformation smoothness.

random_elastic = tio.RandomElasticDeformation(
    max_displacement=max_displacement,
    num_control_points=7,
)
slice_large_displacement_more_cp = random_elastic(slice_grid)
to_pil(slice_large_displacement_more_cp)

#Let's look at the effect of using few control points
#but very large displacements.
random_elastic = tio.RandomElasticDeformation(
    max_displacement=2 * np.array(max_displacement), seed=0,
)
slice_large_displacement = random_elastic(slice_grid)
to_pil(slice_large_displacement)

#Intensity transforms
#Intensity transforms modify only scalar images, whereas label maps are left
#as they were.

#Preprocessing (normalization)
#Rescale intensity
#We can change the intensities range of our images so that it lies within
#e.g. 0 and 1, or -1 and 1, using RescaleIntensity.

rescale = tio.RescaleIntensity((-1, 1))
rescaled = rescale(fpg)
fig, axes = plt.subplots(2, 1)
sns.distplot(fpg.mri.data, ax=axes[0], kde=False)
sns.distplot(rescaled.mri.data, ax=axes[1], kde=False)
axes[0].set_title('Original histogram')
axes[1].set_title('Intensity rescaling')
axes[0].set_ylim((0, 1e6))
axes[1].set_ylim((0, 1e6))
plt.tight_layout()

#There seem to be some outliers with very high intensity.
#We might be able to get rid of those by mapping some percentiles to our final values.

rescale = tio.RescaleIntensity((-1, 1), percentiles=(1, 99))
rescaled = rescale(fpg_ras)
fig, axes = plt.subplots(2, 1)
sns.distplot(fpg.mri.data, ax=axes[0], kde=False)
sns.distplot(rescaled.mri.data, ax=axes[1], kde=False)
axes[0].set_title('Original histogram')
axes[1].set_title('Intensity rescaling with percentiles 1 and 99')
axes[0].set_ylim((0, 1e6))
axes[1].set_ylim((0, 1e6))
plt.tight_layout()
show_fpg(rescaled)

#Z-normalization
#Another common approach for normalization is forcing data points to have zero-mean
#and unit variance. We can use ZNormalization for this.

standardize = tio.ZNormalization()
standardized = standardize(fpg)
fig, axes = plt.subplots(2, 1)
sns.distplot(fpg.mri.data, ax=axes[0], kde=False)
sns.distplot(standardized.mri.data, ax=axes[1], kde=False)
axes[0].set_title('Original histogram')
axes[1].set_title('Z-normalization')
axes[0].set_ylim((0, 1e6))
axes[1].set_ylim((0, 1e6))
plt.tight_layout()

#The second mode in our distribution, corresponding to the foreground, is far from zero
#because the background contributes a lot to the mean computation.
#We can compute the stats using e.g. values above the mean only.
#Let's see if the mean is a good threshold to segment the foreground.

fpg_thresholded = copy.deepcopy(fpg_ras)
data = fpg_thresholded.mri.data
data[data > data.mean()] = data.max()
show_fpg(fpg_thresholded)

#It seems reasonable to use this mask to compute the stats for our normalization transform.

standardize_foreground = tio.ZNormalization(masking_method=lambda x: x > x.mean())
standardized_foreground = standardize_foreground(fpg)
fig, axes = plt.subplots(2, 1)
sns.distplot(fpg.mri.data, ax=axes[0], kde=False)
sns.distplot(standardized_foreground.mri.data, ax=axes[1], kde=False)
axes[0].set_title('Original histogram')
axes[1].set_title('Z-normalization using foreground stats')
axes[0].set_ylim((0, 1e6))
axes[1].set_ylim((0, 1e6))
plt.tight_layout()

#The second mode is now closer to zero, as only the foreground voxels have been
#used to compute the statistics.

#Random blur
#We can use RandomBlur to smooth/blur the images. The standard deviations of the
#Gaussian kernels are expressed in mm and will be computed independently for each axis.

blur = tio.RandomBlur(seed=42)
blurred = blur(fpg_ras)
show_fpg(blurred)

#Random noise
#Gaussian noise can be simulated using RandomNoise. This transform is easiest to use after
#ZNormalization, as we know beforehand that the mean and standard deviation of the input will
#be 0 and 1, respectively. If necessary, the noise mean and std can be set using the
#corresponding keyword arguments.

#Noise in MRI is actually Rician, but it is nearly Gaussian for SNR > 2 (i.e. foreground).

add_noise = tio.RandomNoise(std=0.5, seed=42)
standard = standardize(fpg_ras)
noisy = add_noise(standard)
show_fpg(noisy)

#MRI-specific transforms
#TorchIO includes some transforms to simulate image artifacts specific to MRI modalities.

#Random bias field (DID NOT WORK - REWORK)
#Magnetic field inhomogeneities in the MRI scanner produce low-frequency intensity distortions
#in the images, which are typically corrected using algorithms such as N4ITK.
#To simulate this artifact, we can use RandomBiasField.

#For this example, we will use an image that has been preprocessed so it's meant to be unbiased.

add_bias = tio.RandomBiasField(coefficients=1, seed=0)
mni_bias = add_bias(fpg_ras)
mni_bias.seg = mni_bias.heart
show_fpg(mni_bias)

#k -space transforms
#MR images are generated by computing the inverse Fourier transform of the k-space, which is the signal received by the coils
#in the scanner. If the k-space is altered, an artifact will be created in the image. These artifacts are typically
# accidental, but we can use transforms to simulate them.

#Random spike
#Sometimes, signal peaks can appear in k-space. If one adds a high-energy component at e.g. 440 Hz in the spectrum of
# an audio signal, a tone of that frequency will be audible in the time domain. Similarly, spikes in k-space manifest
# as stripes in image space. They can be simulated using RandomSpike. The number of spikes doesn't affect the transform
# run time, so try adding more!

add_spike = tio.RandomSpike(seed=42)
with_spike = add_spike(fpg_ras)
show_fpg(with_spike)


#Random ghosting
#Ghosting artifacts, caused by patient motion, can be simulated by removing every nth plane from the k-space,
# and can be generated using RandomGhosting. As with the previous transform, the number of ghosts doesn't affect
# the run time, so you can add as many as you like.

add_ghosts = tio.RandomGhosting(intensity=1.8, seed=42)
with_ghosts = add_ghosts(fpg_ras)
show_fpg(with_ghosts)

#Random motion
#If the patient moves during the MRI acquisition, motion artifacts will be present. TorchIO includes an implementation of
# Shaw et al., where the artifact is generated by filling the k-space with random rigidly-transformed versions of the
# original images and computing the inverse transform of the compound k-space.

#Computing the direct and inverse Fourier transform takes some time, so we'll use nearest neighbor interpolation to
# resample faster. Another way of cutting down the run time is using a smaller number of transforms (i.e., the patient
# moves less during acquisition time).

add_motion = tio.RandomMotion(num_transforms=6, image_interpolation='nearest', seed=42)
with_motion = add_motion(fpg_ras)
show_fpg(with_motion)