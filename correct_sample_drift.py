import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
import scipy.ndimage as ndimage
from tifffile import tifffile
import argparse
import os
parser = argparse.ArgumentParser(description='Correct for drifting images in a z stack.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("DATASET_NAME", help="DATASET_NAME")
parser.add_argument("SERVER", help="SERVER")
parser.add_argument("ANALYZER", help="ANALYZER")
args = parser.parse_args()
config = vars(args)
print(config)

DATASET_NAME = config['DATASET_NAME']
SERVER = config['SERVER']
ANALYZER = config['ANALYZER']

data_path = f"/nfs/data{SERVER}/{ANALYZER}/data/{DATASET_NAME}/{DATASET_NAME}_confocal.tif"

images = tifffile.imread(data_path)
n_z = images.shape[0]
n_c = images.shape[1]


which_channel = 1
volume = images[:,which_channel,:,:]
vmax = np.max(volume)

accumulated_shift = np.array([0.0, 0.0])
images_corrected = np.copy(images)

which_channel = 1
ref_volume = images[:,which_channel,:,:]

for i in range(n_z-1):
    image = ref_volume[i,:,:]
    offset_image = ref_volume[i+1,:,:]
    
    # subpixel precision
    shift, error, diffphase = phase_cross_correlation(
        image, offset_image, upsample_factor=100
    )
    
    
    print(f'Detected subpixel offset (y, x): {shift}')
    accumulated_shift += shift

    for i_channel in range(n_c):
        offset_image = images[i+1,i_channel,:,:]
        # corrected_image = fourier_shift(np.fft.fftn(offset_image), accumulated_shift)
        # corrected_image = np.fft.ifftn(corrected_image)
        corrected_image = ndimage.shift(offset_image, accumulated_shift)
        images_corrected[i+1,i_channel,:,:] = corrected_image



save_path = f"/nfs/data{SERVER}/{ANALYZER}/data/{DATASET_NAME}/{DATASET_NAME}_confocal_corrected.tif"
tifffile.imwrite(save_path, images_corrected, shape = images_corrected.shape, planarconfig="separate")