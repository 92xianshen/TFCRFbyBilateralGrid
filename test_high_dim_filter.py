import numpy as np
import PIL.Image as Image
import cv2
from skimage.transform import resize as skresize

from high_dim_filter import HighDimFilter

# ==== Test grayscale joint bilateral upsampling ====

im = Image.open('lena.jpg')
im = np.asarray(im, dtype='float32') / 255.
height, width, n_channels = im.shape[:3]

bilateral_filter = HighDimFilter(is_bilateral=True, height=height, width=width, space_sigma=16., range_sigma=.25)
bilateral_filter.init(im)

inp = im[..., 0]
inp_down = skresize(inp, (height // 4, width // 4))
inp_up = skresize(inp_down, (height, width))
inp_up = inp_up[..., np.newaxis]

result, weight = np.zeros_like(inp_up, dtype=np.float32), np.zeros((height, width, 1), dtype=np.float32)
all_ones = np.ones_like(weight, dtype=np.float32)

result = bilateral_filter.compute(inp_up)
weight = bilateral_filter.compute(all_ones)

result = result / (weight + 1e-5)
result = result.numpy()

cv2.imshow('im', im[..., ::-1])
cv2.imshow('im_down', inp_up)
cv2.imshow('result', result[..., ::-1])
cv2.waitKey()

spatial_filter = HighDimFilter(is_bilateral=False, height=height, width=width, space_sigma=16., range_sigma=.25)
spatial_filter.init(None)

inp = im[..., 0]
inp_down = skresize(inp, (height // 4, width // 4))
inp_up = skresize(inp_down, (height, width))
inp_up = inp_up[..., np.newaxis]

result, weight = np.zeros_like(inp_up, dtype=np.float32), np.zeros((height, width, 1), dtype=np.float32)
all_ones = np.ones_like(weight, dtype=np.float32)

result = spatial_filter.compute(inp_up)
weight = spatial_filter.compute(all_ones)

result = result / (weight + 1e-5)
result = result.numpy()

cv2.imshow('im', im[..., ::-1])
cv2.imshow('im_down', inp_up)
cv2.imshow('result', result[..., ::-1])
cv2.waitKey()