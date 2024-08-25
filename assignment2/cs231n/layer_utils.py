from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# pass
def affine_norm_relu_dp_forward(x, w, b, gamma, beta, bn_param, ln_param, dp_param, norm=None, use_dp=False):
    out, fc_cache, norm_cache, relu_cache, dp_cache = None, None, None, None, None
    out, fc_cache = affine_forward(x, w, b)
    if norm == 'batchnorm':    
        out, norm_cache = batchnorm_forward(out, gamma, beta, bn_param)
    elif norm == 'layernorm':
        out, norm_cache = layernorm_forward(out, gamma, beta, ln_param)
    out, relu_cache = relu_forward(out)
    if use_dp:
        out, dp_cache = dropout_forward(out, dp_param)
    cache = (fc_cache, norm_cache, relu_cache, dp_cache)
    return out, cache

def affine_norm_relu_dp_backward(dout, cache, norm=None, use_dp=False):
    fc_cache, norm_cache, relu_cache, dp_cache = cache
    if use_dp:
        dout = dropout_backward(dout, dp_cache)
    dout = relu_backward(dout, relu_cache)
    if norm is None:
        # dout, dw, db = affine_relu_backward(dout, cache=(fc_cache, relu_cache))
        dgamma, dbeta = 0, 0 
    elif norm == 'batchnorm':
        # dout = relu_backward(dout, relu_cache)
        dout, dgamma, dbeta = batchnorm_backward_alt(dout, norm_cache)
        # dout, dw, db = affine_backward(dout, fc_cache)
    elif norm == 'layernorm':
        # dout = relu_backward(dout, relu_cache)
        dout, dgamma, dbeta = layernorm_backward(dout, norm_cache)
        # dout, dw, db = affine_backward(dout, fc_cache)
    dout, dw, db = affine_backward(dout, fc_cache)
    return dout, dw, db, dgamma, dbeta

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
