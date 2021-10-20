import numpy as np
import PIL.Image as Image
import cv2
from skimage.transform import resize as skresize

from high_dim_filter import BilateralHighDimFilter
import tensorflow as tf

# ==== Test grayscale joint bilateral upsampling ====

im = Image.open('lena.jpg')
im = np.asarray(im, dtype='float32') / 255.
height, width, n_channels = im.shape[:3]

bilateral_filter = BilateralHighDimFilter(height=height, width=width, space_sigma=16., range_sigma=.25)
bilateral_filter.init(tf.constant(im))

inp = im[..., 0]
inp_down = skresize(inp, (height // 4, width // 4))
inp_up = skresize(inp_down, (height, width))
inp_up = inp_up[..., np.newaxis]
print(inp_up.dtype, inp_up.shape, inp_up.max(), inp_up.min())

result, weight = np.zeros_like(inp_up, dtype=np.float32), np.zeros((height, width), dtype=np.float32)
all_ones = np.ones_like(weight, dtype=np.float32)

@tf.function
def ch_filter(inp_ch):
    out_ch = bilateral_filter.compute(inp_ch)
    return out_ch

for ch in range(inp_up.shape[-1]):
    inp_ch = tf.constant(inp_up[..., ch])
    out_ch = ch_filter(inp_ch)
    result[..., ch] = out_ch.numpy()

all_ones_tf = tf.constant(all_ones)
weight_tf = ch_filter(all_ones_tf)
weight[:] = weight_tf.numpy()

result /= (weight[..., np.newaxis] + 1e-10)

cv2.imshow('im', im[..., ::-1])
cv2.imshow('im_down', inp_up)
cv2.imshow('result', result[..., ::-1])
cv2.waitKey()

# spatial_filter = SpatialHighDimFilter(height=height, width=width, space_sigma=16., range_sigma=.25)
# spatial_filter.init(None)

# inp = im[..., 0]
# inp_down = skresize(inp, (height // 4, width // 4))
# inp_up = skresize(inp_down, (height, width))
# inp_up = inp_up[..., np.newaxis]

# result, weight = np.zeros_like(inp_up, dtype=np.float32), np.zeros((height, width, 1), dtype=np.float32)
# all_ones = np.ones_like(weight, dtype=np.float32)

# result = spatial_filter.compute(inp_up)
# weight = spatial_filter.compute(all_ones)

# result = result / (weight + 1e-5)
# result = result.numpy()

# cv2.imshow('im', im[..., ::-1])
# cv2.imshow('im_down', inp_up)
# cv2.imshow('result', result[..., ::-1])
# cv2.waitKey()