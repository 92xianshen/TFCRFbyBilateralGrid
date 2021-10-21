import numpy as np
import PIL.Image as Image
import cv2
from skimage.transform import resize as skresize

from high_dim_filter import SpatialHighDimFilter, BilateralHighDimFilter
import tensorflow as tf

# ==== Test grayscale joint bilateral upsampling ====

im = Image.open('lena.jpg')
im = np.asarray(im, dtype='float32') / 255.
height, width, n_channels = im.shape[:3]

# ==== Joint bilateral upsampling ====

bilateral_filter = BilateralHighDimFilter(height=height, width=width, space_sigma=16., range_sigma=.25)
bilateral_filter.init(tf.constant(im))

inp = im[..., 0]
inp_resample = skresize(inp, (height // 4, width // 4))
inp_resample = skresize(inp_resample, (height, width))
inp_resample = inp_resample[..., np.newaxis]
print(inp_resample.dtype, inp_resample.shape, inp_resample.max(), inp_resample.min())

result, weight = np.zeros_like(inp_resample, dtype=np.float32), np.zeros((height, width, 1), dtype=np.float32)
all_ones = np.ones_like(weight, dtype=np.float32)

@tf.function
def filtering(inp):
    out = bilateral_filter.compute(inp)
    return out

result[:] = filtering(tf.constant(inp_resample)).numpy()

all_ones_tf = tf.constant(all_ones)
weight_tf = filtering(all_ones_tf)
weight[:] = weight_tf.numpy()

result /= (weight + 1e-10)

cv2.imshow('im', im[..., ::-1])
cv2.imshow('im_resample', inp_resample)
cv2.imshow('result', result[..., ::-1])
cv2.waitKey()

# spatial_filter = SpatialHighDimFilter(height=height, width=width, space_sigma=16.)
# spatial_filter.init()

# inp = im[..., 0]
# inp_down = skresize(inp, (height // 4, width // 4))
# inp_up = skresize(inp_down, (height, width))
# inp_up = inp_up[..., np.newaxis]

# result, weight = np.zeros_like(inp_up, dtype=np.float32), np.zeros((height, width, 1), dtype=np.float32)
# all_ones = np.ones_like(weight, dtype=np.float32)

# result = spatial_filter.compute(tf.constant(inp_up))
# weight = spatial_filter.compute(tf.constant(all_ones))

# result = result / (weight + 1e-5)
# result = result.numpy()

# cv2.imshow('im', im[..., ::-1])
# cv2.imshow('im_down', inp_up)
# cv2.imshow('result', result[..., ::-1])
# cv2.waitKey()