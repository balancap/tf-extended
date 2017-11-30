# =========================================================================== #
# [2017] - Robik AI Ltd - Paul Balanca
# All Rights Reserved.

# NOTICE: All information contained herein is, and remains
# the property of Robik AI Ltd, and its suppliers
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Robik AI Ltd
# and its suppliers and may be covered by U.S., European and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Robik AI Ltd.
# =========================================================================== #
"""Misc. collection of useful layers, mostly very simple!
"""
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope

slim = tf.contrib.slim

# =========================================================================== #
# Tools...
# =========================================================================== #

def _get_dimension(shape, dim, min_rank=1):
    """Returns the `dim` dimension of `shape`, while checking it has `min_rank`.
    Args:
        shape: A `TensorShape`.
        dim: Integer, which dimension to return.
        min_rank: Integer, minimum rank of shape.
    Returns:
        The value of the `dim` dimension.
    Raises:
        ValueError: if inputs don't have at least min_rank dimensions, or if the
            first dimension value is not defined.
    """
    dims = shape.dims
    if dims is None:
        raise ValueError('dims of shape must be known but is None')
    if len(dims) < min_rank:
        raise ValueError('rank of shape must be at least %d not: %d' % (min_rank,
                                                                        len(dims)))
    value = dims[dim].value
    if value is None:
        raise ValueError(
            'dimension %d of shape must be known but is None: %s' % (dim, shape))
    return value


# =========================================================================== #
# Extension of TensorFlow common layers.
# =========================================================================== #
@add_arg_scope
def channel_dimension(shape, data_format, min_rank=1):
    """Returns the channel dimension of shape, while checking it has min_rank.
    Args:
        shape: A `TensorShape`.
        data_format: `NCHW` or `NHWC`.
        min_rank: Integer, minimum rank of shape.
    Returns:
         value of the first dimension.
    Raises:
        ValueError: if inputs don't have at least min_rank dimensions, or if the
            first dimension value is not defined.
    """
    return _get_dimension(shape, 1 if data_format == 'NCHW' else -1,
                          min_rank=min_rank)

@add_arg_scope
def channel_to_last(inputs, data_format='NHWC', scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a consistent NHWC output format whatever the input data format.

    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'channel_to_last', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return net


@add_arg_scope
def concat_channels(l_inputs, data_format='NHWC', scope=None):
    """Concat a list of tensors on the channel axis.

    Args:
      inputs: List Tensors;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'concat_channels', l_inputs):
        if data_format == 'NHWC':
            net = tf.concat(l_inputs, axis=3)
        elif data_format == 'NCHW':
            net = tf.concat(l_inputs, axis=1)
        return net

@add_arg_scope
def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          data_format='NHWC',
          scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.

    Aims to mimic padding in Caffe and MXNet, helping the port of models to
    TensorFlow. Tries to follow the naming convention of `tf.contrib.layers`.

    Args:
      inputs: 4D input Tensor;
      pad: 2-Tuple with padding values for H and W dimensions;
      mode: Padding mode. C.f. `tf.pad`
      data_format:  NHWC or NCHW data format.
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        # Padding shape.
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net

@add_arg_scope
def pad_logits(logits, pad=(0, 0)):
    """Pad logits Tensor, to deal with different number of classes.
    """
    shape = logits.get_shape().as_list()
    dtype = logits.dtype
    l = [logits]
    if pad[0] > 0:
        a = tf.constant(dtype.min, dtype, (shape[0], pad[0]))
        l = [a] + l
    if pad[1] > 0:
        a = tf.constant(dtype.min, dtype, (shape[0], pad[1]))
        l = l + [a]
    output = tf.concat(l, axis=1)
    return output

@add_arg_scope
def spatial_mean(inputs, scaling=None, keep_dims=False,
                 data_format='NHWC', scope=None):
    """Average tensor along spatial dimensions.

    Args:
      inputs: Input tensor;
      keep_dims: Keep spatial dimensions?
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'spatial_mean', [inputs]):
        axes = [1, 2] if data_format == 'NHWC' else [2, 3]
        net = tf.reduce_mean(inputs, axes, keep_dims=keep_dims)
        return net

@add_arg_scope
def spatial_squeeze(inputs, data_format='NHWC', scope=None):
    """Squeeze spatial dimensions, if possible.

    Args:
      inputs: Input tensor;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'spatial_squeeze', [inputs]):
        axes = [1, 2] if data_format == 'NHWC' else [2, 3]
        net = tf.squeeze(inputs, axes)
        return net

@add_arg_scope
def ksize_for_squeezing(inputs, default_ksize=1024, data_format='NHWC'):
    """Get the correct kernel size for squeezing an input tensor.
    """
    shape = inputs.get_shape().as_list()
    kshape = shape[1:3] if data_format == 'NHWC' else shape[2:]
    if kshape[0] is None or kshape[1] is None:
        kernel_size_out = [default_ksize, default_ksize]
    else:
        kernel_size_out = [min(kshape[0], default_ksize),
                           min(kshape[1], default_ksize)]
    return kernel_size_out

@add_arg_scope
def batch_norm(inputs,
               activation_fn=None,
               normalizer_fn=None,
               normalizer_params=None):
    """Batch normalization layer compatible with the classic conv. API.
    Simpler to use with arg. scopes.
    """
    outputs = inputs
    # BN...
    if normalizer_fn is not None:
        normalizer_params = normalizer_params or {}
        outputs = normalizer_fn(outputs, **normalizer_params)
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs

