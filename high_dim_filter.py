'''
High-dimensional filter implemented in TF 2.x
Spatial high-dim filter if `features` is `None`
Bilateral high-dim filter otherwise
'''

import numpy as np
import itertools as it
import tensorflow as tf

def clamp(min_value: tf.float32, max_value: tf.float32, x: tf.Tensor) -> tf.Tensor:
    return tf.maximum(min_value, tf.minimum(max_value, x))

class HighDimFilter:
    '''
    High-dimensional filter
    '''
    def __init__(self, is_bilateral: bool, height: int, width: int, space_sigma: float=16, range_sigma: float=.25, padding_xy: int=2, padding_z: int=2) -> None:
        '''
        Initialization

        Args:
            height: image height, int
            width: image width, int
            space_sigma: sigma_s, float
            range_sigma: sigma_r, float
            padding_xy: number of pixel for padding along y and x, int
            padding_z: number of pixel for padding along z

        Returns:
            None
        '''
        # Is bilateral?
        self.is_bilateral = is_bilateral
        # Index order: y --> height, x --> width, z --> depth
        self.height, self.width = height, width
        self.space_sigma, self.range_sigma = space_sigma, range_sigma
        self.padding_xy, self.padding_z = padding_xy, padding_z

    def init(self, features: tf.Tensor=None) -> None:
        # Initialize a spatial high-dim filter if `features` is None; otherwise a bilateral one and `features` should be three-channel and channel-last
        if self.is_bilateral:
            tf.assert_equal(tf.shape(features)[-1], 3)
            tf.assert_equal(features.ndim, 3)

        # Height and width of data grid, scala, dtype int
        self.small_height = tf.cast(tf.cast(self.height - 1, dtype=tf.float32) / self.space_sigma, dtype=tf.int32) + 1 + 2 * self.padding_xy
        self.small_width = tf.cast(tf.cast(self.width - 1, dtype=tf.float32) / self.space_sigma, dtype=tf.int32) + 1 + 2 * self.padding_xy

        # Space coordinates, shape (h, w), dtype int
        yy, xx = tf.meshgrid(tf.range(self.height), tf.range(self.width), indexing='ij') # (h, w)
        yy, xx = tf.cast(yy, dtype=tf.float32), tf.cast(xx, dtype=tf.float32)
        # Spatial coordinates of splat, shape (h, w)
        splat_yy = tf.cast(yy / self.space_sigma + .5, tf.int32) + self.padding_xy
        splat_xx = tf.cast(xx / self.space_sigma + .5, tf.int32) + self.padding_xy
        # Spatial coordinates of slice, shape (h, w)
        slice_yy = tf.cast(yy, tf.float32) / self.space_sigma + self.padding_xy
        slice_xx = tf.cast(xx, tf.float32) / self.space_sigma + self.padding_xy

        # Method to get left and right indices of slice interpolation
        def get_both_indices(size, coord):
            left_index = clamp(0, size - 1, tf.cast(coord, tf.int32))
            right_index = clamp(0, size - 1, left_index + 1)
            return left_index, right_index

        # Spatial interpolation index of slice
        y_index, yy_index = get_both_indices(self.small_height, slice_yy) # (h, w)
        x_index, xx_index = get_both_indices(self.small_width, slice_xx) # (h, w)

        # Spatial interpolation factor of slice
        y_alpha = tf.reshape(slice_yy - tf.cast(y_index, dtype=tf.float32), [-1, ]) # (h x w, )
        x_alpha = tf.reshape(slice_xx - tf.cast(x_index, dtype=tf.float32), [-1, ]) # (h x w, )

        if not self.is_bilateral:
            # Shape of spatial data grid
            self.data_shape = [self.small_height, self.small_width]

            # Spatial splat coordinates, shape (h x w, )
            self.splat_coords = splat_yy * self.small_width + splat_xx
            self.splat_coords = tf.reshape(self.splat_coords, [-1, ])

            # Slice interpolation index and factor
            self.interp_indices = [y_index, yy_index, x_index, xx_index] # (4, h x w)
            self.alphas = [1. - y_alpha, y_alpha, 1. - x_alpha, x_alpha] # (4, h x w)

            # Spatial convolutional dimension
            self.dim = 2
        else:
            # Decompose `features` into r, g, and b channels
            r, g, b = features[..., 0], features[..., 1], features[..., 2]
            r_min, r_max = tf.reduce_min(r), tf.reduce_max(r)
            g_min, g_max = tf.reduce_min(g), tf.reduce_max(g)
            b_min, b_max = tf.reduce_min(b), tf.reduce_max(b)
            r_delta, g_delta, b_delta = r_max - r_min, g_max - g_min, b_max - b_min
            # Range coordinates, shape (h, w), dtype float
            rr, gg, bb = r - r_min, g - g_min, b - b_min

            # Depths of data grid
            self.small_rdepth = int(r_delta / self.range_sigma) + 1 + 2 * self.padding_z
            self.small_gdepth = int(g_delta / self.range_sigma) + 1 + 2 * self.padding_z
            self.small_bdepth = int(b_delta / self.range_sigma) + 1 + 2 * self.padding_z

            # Range coordinates, shape (h, w)
            splat_rr = tf.cast(rr / self.range_sigma + .5, tf.int32) + self.padding_z
            splat_gg = tf.cast(gg / self.range_sigma + .5, tf.int32) + self.padding_z
            splat_bb = tf.cast(bb / self.range_sigma + .5, tf.int32) + self.padding_z

            # Range coordinates, shape (h, w)
            slice_rr = rr / self.range_sigma + self.padding_z
            slice_gg = gg / self.range_sigma + self.padding_z
            slice_bb = bb / self.range_sigma + self.padding_z

            # Slice interpolation range coordinate pairs
            r_index, rr_index = get_both_indices(self.small_rdepth, slice_rr) # (h, w)
            g_index, gg_index = get_both_indices(self.small_gdepth, slice_gg) # (h, w)
            b_index, bb_index = get_both_indices(self.small_bdepth, slice_bb) # (h, w)

            # Interpolation factors
            r_alpha = tf.reshape(slice_rr - tf.cast(r_index, dtype=tf.float32), [-1, ]) # (h x w, ) 
            g_alpha = tf.reshape(slice_gg - tf.cast(g_index, dtype=tf.float32), [-1, ]) # (h x w, )
            b_alpha = tf.reshape(slice_bb - tf.cast(b_index, dtype=tf.float32), [-1, ]) # (h x w, )

            # Shape of bilateral data grid
            self.data_shape = [self.small_height, self.small_width, self.small_rdepth, self.small_gdepth, self.small_bdepth]

            # Bilateral splat coordinates, shape (h x w, )
            self.splat_coords = (((splat_yy * self.small_width + splat_xx) * self.small_rdepth + splat_rr) * self.small_gdepth + splat_gg) * self.small_bdepth + splat_bb
            self.splat_coords = tf.reshape(self.splat_coords, [-1, ]) # (h x w, )

            # Bilateral interpolation index and factor
            self.interp_indices = [y_index, yy_index, x_index, xx_index, r_index, rr_index, g_index, gg_index, b_index, bb_index] # (10, h x w)
            self.alphas = [1. - y_alpha, y_alpha, 1. - x_alpha, x_alpha, 1. - r_alpha, r_alpha, 1. - g_alpha, g_alpha, 1. - b_alpha, b_alpha] # (10, h x w)

            # Bilateral convolutional dimension
            self.dim = 5

    def convn(self, data, n_iter: int=2) -> tf.Tensor:
        buffer = tf.zeros_like(data)
        perm = list(range(1, data.ndim)) + [0] # [1, ..., ndim - 1, 0]

        for _ in range(n_iter):
            buffer, data = data, buffer

            for dim in range(data.ndim):
                newdata = (buffer[:-2] + buffer[2:] + 2 * buffer[1:-1]) / 4.
                data = tf.concat([data[:1], newdata, data[-1:]], axis=0)
                data = tf.transpose(data, perm=perm)
                buffer = tf.transpose(buffer, perm=perm)

        del buffer
        return data

    def loop_Nlinear_interpolation(self, data: tf.Tensor) -> tf.Tensor:
        # Method of coordinate transformation
        def set_coord_transform():
            def bilateral_coord_transform(y_idx, x_idx, r_idx, g_idx, b_idx):
                return tf.reshape((((y_idx * self.small_width + x_idx) * self.small_rdepth + r_idx) * self.small_gdepth + g_idx) * self.small_bdepth + b_idx, [-1, ]) # (h x w, )

            def spatial_coord_transform(y_idx, x_idx):
                return tf.reshape(y_idx * self.small_width + x_idx, [-1, ]) # (h x w, )

            if self.is_bilateral:
                return bilateral_coord_transform
            else:
                return spatial_coord_transform

        coord_transform = set_coord_transform()

        # Initialize interpolation
        interpolation = tf.zeros(shape=[self.height * self.width, ], dtype=tf.float32)
        offset = tf.range(self.dim) * 2
        
        for perm in it.product(range(2), repeat=self.dim):
            alpha_prod = tf.gather(self.alphas, tf.constant(perm) + offset)
            idx = tf.gather(self.interp_indices, tf.constant(perm) + offset)

            data_slice = tf.gather(tf.reshape(data, [-1, ]), coord_transform(*idx))
            interpolation = interpolation + tf.math.reduce_prod(alpha_prod, axis=0) * data_slice

        return interpolation

    def compute(self, inp: tf.Tensor) -> tf.Tensor:
        # `inp` should be 3-dim
        tf.assert_equal(inp.ndim, 3)
        # Channel-last to channel-first because tf.map_fn
        inp_transpose = tf.transpose(inp, (2, 0, 1))

        def ch_filter(inp_ch: tf.Tensor) -> tf.Tensor:
            # Filter each channel
            inp_flat = tf.reshape(inp_ch, [-1, ]) # (h x w)
            # ==== Splat ====
            data_flat = tf.math.bincount(self.splat_coords, weights=inp_flat, minlength=tf.math.reduce_prod(self.data_shape), maxlength=tf.math.reduce_prod(self.data_shape), dtype=tf.float32)
            data = tf.reshape(data_flat, shape=self.data_shape)
            
            # ==== Blur ====
            data = self.convn(data)

            # ==== Slice ====
            interpolation = self.loop_Nlinear_interpolation(data)
            interpolation = tf.reshape(interpolation, shape=(self.height, self.width))

            return interpolation

        out_transpose = tf.map_fn(ch_filter, inp_transpose)
        out = tf.transpose(out_transpose, (1, 2, 0)) # Channel-first to channel-last

        return out