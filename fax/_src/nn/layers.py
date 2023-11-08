# from collections.abc import Callable
import functools
from typing import Optional, Sequence, Tuple, Union, Callable

import chex
import jax
import jax.numpy as jnp
from jax.nn import normalize
import numpy as np
import operator as op
from jax import lax
from jax import nn
from jax import random
from jax.tree_util import tree_map

import fax.nn.functions as fax_funcs
import fax.nn.initializers as fax_inits
from fax.rl.types import ObsType

Array = list[chex.Array]
Scalar = chex.Scalar
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
Shape = tuple[int, ...]
Initializer = Callable[[PRNGKey, Shape, jnp.dtype], Array]

InitOutType = tuple[str, Shape, dict, dict]
InitInType = tuple[PRNGKey, Shape, dict]
InitFunc = Callable[InitInType, InitOutType]

ApplyInType = tuple[dict, dict, dict]
ApplyOutType = tuple[dict, dict]
ApplyLayerFunc = Callable[ApplyInType, ApplyOutType]
ApplyHebbianFunc = Callable[[dict, Array, Array, Array], Array]
ActivationFunc = Callable[Array, Array]

Layer = tuple[InitFunc, ApplyLayerFunc]
Layers = tuple[Layer, ...]
LocalLayer = tuple[InitFunc, ApplyHebbianFunc]

DispatchFunc = Union[str, Callable]


def dropout(rate: float) -> Layer:
    """Layer construction function for a dropout layer with given rate."""

    def init_func(rng, init_params: dict, init_states: dict, input_shape):
        # mode is 0 if train 
        states = {"mode": 0}
        return {}, {}, states, input_shape

    def apply_func(rng: PRNGKey, params: dict, states: dict, inp):
        mode = states["mode"]
        output = jax.lax.select(mode == 0, 
                                jnp.where(
                                    random.bernoulli(rng, rate, inp.shape), 
                                    inp / rate, 0.0),
                                inp)
        
        return states, output

    return init_func, apply_func

def application(application_type: DispatchFunc = "linear"):
    func = fax_funcs.get(application_type)
    
    def init_fun(rng, init_params: dict, 
                 init_states: dict, input_shape: Shape):
        return {}, {}, {}, input_shape
    
    def apply_func(rng: PRNGKey, params: dict, states: dict, inp):
        out = func(inp)
        return {}, out
    return init_fun, apply_func
        
        
def frozen(layer: Layer) -> Layer:
    init_fun, apply_fun = layer

    def frozen_apply_fun(rng, params: dict, *args, **kwargs):
        params = tree_map(lambda x: lax.stop_gradient(x), params)
        return apply_fun(rng, params, *args, **kwargs)

    return init_fun, frozen_apply_fun

def frozen_input() -> Layer:
    def init_fun(rng, init_params: dict,
                 init_states: dict, input_shape: Shape):
        return {}, {}, {}, input_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        inp = tree_map(lambda x: lax.stop_gradient(x), inp)
        return {}, inp 

    return init_fun, apply_fun

def empty() -> Layer:
    def init_fun(rng, init_params: dict,
                 init_states: dict, input_shape: Shape):
        return {}, {}, {}, ()

    def apply_fun(rng: PRNGKey, params: dict, states: dict, *args):
        return {}, ()

    return init_fun, apply_fun


def general_conv(dimension_numbers, out_chan, filter_shape,
                 strides=None, padding='VALID',
                 use_bias: bool = False,
                 w_init: DispatchFunc = "glorot_normal",
                 b_init: DispatchFunc = "normal"):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    w_init = fax_inits.get(w_init)
    b_init = fax_inits.get(b_init)()
    strides = strides or one
    w_init = w_init(rhs_spec.index('I'), rhs_spec.index('O'))
    
    def init_fun(rng, init_params: dict,
                 init_states: dict, input_shape: Shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        input_shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers)
        bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
        k1, k2 = random.split(rng)
        params = {"w": w_init(k1, kernel_shape), "b": b_init(k2, bias_shape)}
        return {}, params, {}, output_shape
    
    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        is_zero = jnp.all(inp == 0.0)
        w, b = params["w"], params["b"]
        output = lax.conv_general_dilated(
            inp, w, strides, padding, one, one,
            dimension_numbers=dimension_numbers)
        if use_bias:
            output += b
        return {}, (1 - is_zero) * output
    return init_fun, apply_fun

    """
    assumed dataformat for (input, kernel, output) are:
    0D ('NC', 'IO', 'NC')
    1D ('NHC', 'HIO', 'NHC')
    2D ('NHWC', 'HWIO', 'NHWC')
    3D ('NHWDC', 'HWDIO', 'NHWDC')
    """


conv_0D = functools.partial(general_conv, ('NC', 'IO', 'NC'))
conv_1D = functools.partial(general_conv, ('NHC', 'HIO', 'NHC'))
conv_2D = functools.partial(general_conv, ('NHWC', 'HWIO', 'NHWC'))
conv_3D = functools.partial(general_conv, ('NHWDC', 'HWDIO', 'NHWDC'))


def _pooling_layer(reducer, init_val, rescaler=None):
    def PoolingLayer(window_shape, strides=None, padding='VALID', spec=None):
        """Layer construction function for a pooling layer."""
        strides = strides or (1,) * len(window_shape)
        rescale = rescaler(window_shape, strides, padding) if rescaler else None

        if spec is None:
            non_spatial_axes = 0, len(window_shape) + 1
        else:
            non_spatial_axes = spec.index('N'), spec.index('C')

        for i in sorted(non_spatial_axes):
            window_shape = window_shape[:i] + (1,) + window_shape[i:]
            strides = strides[:i] + (1,) + strides[i:]

        def init_fun(rng, init_params: dict, init_states: dict, input_shape: Shape):
            padding_vals = lax.padtype_to_pads(input_shape, window_shape,
                                               strides, padding)
            ones = (1,) * len(window_shape)
            out_shape = lax.reduce_window_shape_tuple(
                input_shape, window_shape, strides, padding_vals, ones, ones)
            return {}, {}, {}, out_shape
        
        
        def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
            out = lax.reduce_window(inp, init_val, reducer, window_shape,
                                    strides, padding)
            out = rescale(out, inp, spec) if rescale else out
            return {}, out
        
        return init_fun, apply_fun
    return PoolingLayer


max_pool = _pooling_layer(lax.max, -jnp.inf)
sum_pool = _pooling_layer(lax.add, 0.)


def _normalize_by_window_size(dims, strides, padding):
    def rescale(outputs, inputs, spec):
        if spec is None:
            non_spatial_axes = 0, inputs.ndim - 1
        else:
            non_spatial_axes = spec.index('N'), spec.index('C')

        spatial_shape = tuple(inputs.shape[i] for i in range(inputs.ndim)
                              if i not in non_spatial_axes)
        one = jnp.ones(spatial_shape, dtype=inputs.dtype)
        window_sizes = lax.reduce_window(
            one, 0., lax.add, dims, strides, padding)
        for i in sorted(non_spatial_axes):
            window_sizes = jnp.expand_dims(window_sizes, i)
        return outputs / window_sizes
    return rescale


avg_pool = _pooling_layer(lax.add, 0., _normalize_by_window_size)


def flatten():
    """Layer construction function for flattening all but the leading dim."""
    def init_fun(rng, init_params: dict, 
                 init_states: dict, input_shape: Shape):
        output_shape = (
            input_shape[0], functools.reduce(op.mul, input_shape[1:], 1))
        return {}, {}, {}, output_shape
   
    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        return {}, jnp.reshape(inp, (inp.shape[0], -1))
    return init_fun, apply_fun

def batch_norm(axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True,
              beta_init: DispatchFunc = "zeros", 
              gamma_init: DispatchFunc = "ones"):
    beta_init = fax_inits.get(beta_init)
    gamma_init = fax_inits.get(gamma_init)
    """Layer construction function for a batch normalization layer."""
    _beta_init = lambda rng, shape: beta_init(rng, shape) if center else ()
    _gamma_init = lambda rng, shape: gamma_init(rng, shape) if scale else ()
    axis = (axis,) if jnp.isscalar(axis) else axis
   
    def init_fun(rng, init_params: dict, 
                 init_states: dict, input_shape: Shape):
        shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
        k1, k2 = random.split(rng)
        beta, gamma = _beta_init(k1, shape), _gamma_init(k2, shape)
        params = {"beta": beta, "gamma": gamma}
        return {}, params, {}, input_shape
    
    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        beta, gamma = params["beta"], params["gamma"]
        # TODO(phawkins): jnp.expand_dims should accept an axis tuple.
        # (https://github.com/numpy/numpy/issues/12290)
        ed = tuple(
            None if i in axis else slice(None) for i in range(jnp.ndim(inp)))
        z = normalize(inp, axis, epsilon=epsilon)
        if center and scale:
            return {}, gamma[ed] * z + beta[ed]
        if center:
            return {}, z + beta[ed]
        if scale:
            return gamma[ed] * z
        return {}, z
    return init_fun, apply_fun

def general_dilated_conv(filters: int,
                         kernel_size: Sequence[int],
                         window_strides: Sequence[int],
                         padding: Union[str, Sequence[Tuple[int, int]]],
                         input_dilation: Optional[Sequence[int]],
                         kernel_dilation: Optional[Sequence[int]],
                         activation: DispatchFunc = "linear",
                         w_init: DispatchFunc = "orthogonal",
                         b_init: DispatchFunc = "zeros",
                         feature_group_count: int = 1,
                         batch_group_count: int = 1) -> Layer:
    """
    assumed dataformat for (input, kernel, output) are:
    0D ('NC', 'IO', 'NC')
    1D ('NHC', 'HIO', 'NHC')
    2D ('NHWC', 'HWIO', 'NHWC')
    3D ('NHWDC', 'HWDIO', 'NHWDC')
    """
    activation = fax_funcs.get(activation)
    w_init = fax_inits.get(w_init)
    b_init = fax_inits.get(b_init)

    def init_func(rng, init_params: dict, init_states: dict,
                  input_shape: Shape):
        ndim = len(input_shape)
        if ndim == 2:
            dimension_numbers = ('NC', 'IO', 'NC')
        if ndim == 3:
            dimension_numbers = ('NHC', 'HIO', 'NHC')
        if ndim == 4:
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        if ndim == 5:
            dimension_numbers = ('NHWDC', 'HWDIO', 'NHWDC')
        kernel_shape = tuple(kernel_size) + (filters,)
        w = w_init(kernel_shape)
        dn = lax.conv_dimension_numbers(input_shape,
                                        kernel_shape,
                                        dimension_numbers)
        # use dummy input in order to compute output shape
        dummy_in = jnp.ones(input_shape, dtype=jnp.float32)
        dummpy_out = lax.conv_general_dilated(dummy_in, w,
                                              window_strides, padding,
                                              input_dilation,
                                              kernel_dilation,
                                              dn,
                                              feature_group_count,
                                              batch_group_count)
        output_shape = jnp.shape(dummpy_out)
        bias_shape = output_shape[1:]
        b = b_init(bias_shape)
        return {}, {"w": w, "b": b}, {"dn": dn}, output_shape

    def apply_func(rng: PRNGKey, params: dict, states: dict, inp):
        is_zero = jnp.all(inp == 0.0)
        w = params["w"]
        b = params["b"]
        dn = states["dn"]
        out = lax.conv_general_dilated(inp, w,
                                       window_strides, padding, input_dilation,
                                       kernel_dilation, dn,
                                       feature_group_count, batch_group_count)
        out = activation(out + b)
        out = (1 - is_zero) * out
        return {"dn": dn}, out
    return init_func, apply_func


def projection(axis) -> Layer:
    """
    Helper function that project a tuple of inputs to a sub-collection of it.
    :param axis: a int, or Sequence[int]
    :return:
    """
    # here we prevent to make a tuple with one element is axis is a int
    if not isinstance(axis, int):
        axis = jnp.array(axis, dtype=jnp.int32)
        resolve_axis = lambda arr: tuple([arr[i] for i in axis])
    else:
        resolve_axis = lambda arr: arr[axis]

    def init_func(rng, init_params: dict, init_states: dict, input_shape):
        return {}, {}, {}, resolve_axis(input_shape)

    def apply_func(rng: PRNGKey, params: dict, states: dict, inp):
        return {}, resolve_axis(inp)

    return init_func, apply_func


def identity() -> Layer:
    def init_func(rng, init_params: dict, init_states: dict, input_shape):
        return {}, {}, {}, input_shape

    def apply_func(rng: PRNGKey, params: dict, states: dict, inp):
        return {}, inp

    return init_func, apply_func


def summation():
    """
    layer that produce a summation of inputs,
    each inputs layers need to have the same output dimension
    :return:
    """

    def init_func(rng: PRNGKey, init_params: dict,
                  init_states: dict, *inputs_layers):
        output_shape = inputs_layers[0]
        return {}, {}, {}, output_shape

    def apply_func(rng: PRNGKey, params: dict, states: dict, *inputs_layers):
        layers_sum = jnp.sum(jnp.array(list(inputs_layers)), axis=0)
        return {}, layers_sum

    return init_func, apply_func


def concat(axis: int = -1):
    """
    Concatenation of multiple layers output over one axis.
    vector must have the same number of dimensions
    and the same dimensions except possibly for the axis dimension
    :param name:
    :param axis:
    :return: a layer that concatenate output of layers
    """

    def init_func(rng, init_params: dict, init_states: dict,
                  *inputs_layers):
        layers_sum = jnp.sum(jnp.array(list(inputs_layers)), axis=0)
        local_axis = len(layers_sum) - 1 if axis == -1 else axis
        new_dim = layers_sum[local_axis]
        shapes = inputs_layers[0]
        output_shape = tuple(shapes[0:local_axis]) +\
            (int(new_dim),) + tuple(shapes[local_axis + 1:])
        return {}, {}, {}, output_shape

    def apply_func(rng: PRNGKey, params: dict, states: dict, *inputs_layers):
        concat_output = jnp.concatenate(list(inputs_layers), axis=axis)
        return {}, concat_output

    return init_func, apply_func


def dense(out_dim: int, activation: DispatchFunc = "linear",
          is_feedback_aligned=False,
          w_init: DispatchFunc = "orthogonal",
          b_init: DispatchFunc = "zeros"):
    """Layer constructor function for a dense (fully-connected) layer."""
    activation = fax_funcs.get(activation)
    w_init = fax_inits.get(w_init)()
    b_init = fax_inits.get(b_init)

    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        output_shape = input_shape[:-1] + (out_dim,) \
            if len(input_shape) > 1 else (out_dim,)
        k1, k2, k3 = random.split(rng, 3)
        w_ff, b_ff = w_init(k1, (input_shape[-1], out_dim), jnp.float32),\
            b_init(k2, (out_dim,), jnp.float32)
        params = {"w_ff": w_ff, "b_ff": b_ff}
        if is_feedback_aligned:
            w_fb = w_init(k1, (out_dim, input_shape[-1]))
            params["w_fb"] = w_fb

        return {}, params, {}, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        is_zero = jnp.all(inp == 0.0)
        w, b = params["w_ff"], params["b_ff"]
        output = activation(jnp.dot(inp, w) + b)
        output = (1 - is_zero) * output
        return {}, output

    @jax.custom_vjp
    def apply_fun_fa(rng: PRNGKey, params: dict, states: dict, inp):
        is_zero = jnp.all(inp == 0.0)
        w, b = params["w_ff"], params["b_ff"]
        output = activation(jnp.dot(inp, w) + b)
        output = (1 - is_zero) * output
        return {}, output

    def f_fwd(rng: PRNGKey, params: dict, states: dict, inp):
        # Returns primal output
        # and residuals to be used in backward pass by f_bwd.
        w_ff, b_ff = params["w_ff"], params["b_ff"]
        w_fb = params["w_fb"]

        return apply_fun_fa(rng, params, states, inp), (inp, w_ff, b_ff, w_fb)

    def f_bwd(res, g):
        inp, w_ff, b_ff, w_fb = res
        d_e_d_y = g[1]
        a = jnp.dot(inp, w_ff) + b_ff
        d_f_d_a = jax.vmap(jax.grad(activation))(a)
        d_e_d_a = d_e_d_y * d_f_d_a
        d_w_ff = d_e_d_a * jnp.expand_dims(inp, 1)
        d_b_ff = d_e_d_a
        d_w_fb = inp * jnp.expand_dims(d_e_d_y, 1)
        d_x = jnp.dot(d_e_d_y, w_fb)
        return None, {"w_ff": d_w_ff, "b_ff": d_b_ff, "w_fb": d_w_fb},\
            None, d_x

    apply_fun_fa.defvjp(f_fwd, f_bwd)
    apply_fun = apply_fun_fa if is_feedback_aligned else apply_fun

    return init_fun, apply_fun


def gaussian(out_dim: int, activation: DispatchFunc = "linear",
             mu_kernel_init: DispatchFunc = "orthogonal",
             mu_bias_init: DispatchFunc = "zeros",
             sigma_kernel_init: DispatchFunc = "orthogonal",
             sigma_bias_init: DispatchFunc = "zeros"):
    """Layer constructor function for a dense (fully-connected) layer."""
    activation = fax_funcs.get(activation)
    mu_kernel_init = fax_inits.get(mu_kernel_init)()
    mu_bias_init = fax_inits.get(mu_bias_init)
    sigma_kernel_init = fax_inits.get(sigma_kernel_init)()
    sigma_bias_init = fax_inits.get(sigma_bias_init)

    def init_fun(rng: PRNGKey, init_params: dict, 
                 init_states: dict, input_shape):
        output_shape = input_shape[:-1] + (out_dim,) \
            if len(input_shape) > 1 else (out_dim,)
        k1, k2, k3, k4 = random.split(rng, 4)
        mu_w, mu_b, sigma_w, sigma_b = \
            mu_kernel_init(k1, (input_shape[-1], out_dim), jnp.float32), \
            mu_bias_init(k2, (out_dim,), jnp.float32), \
            sigma_kernel_init(k3, (input_shape[-1], out_dim), jnp.float32), \
            sigma_bias_init(k4, (out_dim,), jnp.float32),

        params = {"mu_w": mu_w, "mu_b": mu_b,
                  "sigma_w": sigma_w, "sigma_b": sigma_b}

        return {}, params, {}, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        is_zero = jnp.all(inp == 0.0)
        mu_w, mu_b, sigma_w, sigma_b = params["mu_w"], params["mu_b"],\
            params["sigma_w"], params["sigma_b"]
        mean = activation(jnp.dot(inp, mu_w) + mu_b)
        sigma = nn.relu(jnp.dot(inp, sigma_w) + sigma_b)
        output = (1 - is_zero) * (
            mean + sigma * jax.random.normal(
                rng, (out_dim,), dtype=jnp.float32))
        return {}, output

    return init_fun, apply_fun

def dirichlet(out_dim: int, activation: DispatchFunc = "relu",
              concentration_kernel_init: DispatchFunc = "orthogonal"):
    # mu_bias_init: DispatchFunc = "zeros",
    # sigma_kernel_init: DispatchFunc = "orthogonal",
    # sigma_bias_init: DispatchFunc = "zeros"):
    """Layer constructor function for a dense (fully-connected) layer."""
    
    activation = fax_funcs.get(activation)
    c_kernel_init = fax_inits.get(concentration_kernel_init)()

    # mu_bias_init = fax_inits.get(mu_bias_init)
    # sigma_kernel_init = fax_inits.get(sigma_kernel_init)
    # sigma_bias_init = fax_inits.get(sigma_bias_init)

    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        output_shape = (out_dim,)
        k1, k2, k3, k4 = random.split(rng, 4)
        # mu_w, mu_b, sigma_w, sigma_b = \
        #     mu_kernel_init(k1, (input_shape[-1], out_dim), jnp.float32), \
        #     mu_bias_init(k2, (out_dim,), jnp.float32),\
        #     sigma_kernel_init(k3, (input_shape[-1], out_dim), jnp.float32),\
        #     sigma_bias_init(k4, (out_dim,), jnp.float32)
        c_w = c_kernel_init(k1, (input_shape[-1], out_dim), jnp.float32)
        params = {"c_w": c_w}

        return {}, params, {}, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        is_zero = jnp.all(inp == 0.0)
        c_w = params["c_w"]
        c = activation(jnp.dot(inp, c_w))
        output = (1 - is_zero) * (
            jax.random.dirichlet(rng, c, dtype=jnp.float32))
        return {}, output

    return init_fun, apply_fun

def peephole_lstm(out_dim:int,
                  activation: DispatchFunc = "tanh",
                  recurrent_activation: DispatchFunc = "hard_sigmoid",
                  w_init: DispatchFunc = "glorot_uniform",
                  u_init: DispatchFunc = "orthogonal",
                  b_init: DispatchFunc = "zeros",
                  p_init: DispatchFunc = "normal",
                  use_bias: bool = True,
                  full_cell_reccurent: bool = False,
                  use_activation_at_output: bool = True):
    """ peehole lstm implementation

    Args:
        out_dim (int): number of cell in the lstm unit
        w_init (DispatchFunc, optional): kernel intialization.
        Defaults to "glorot_normal".
        b_init (DispatchFunc, optional): bias initialization. Defaults to "normal".
        full_cell_reccurent (bool, optional): 
        choose between peephole implementation
        false is the vanilia implementation where cell states information only flow
        to their respective gate (LSTM: A search space odyssey, 2017,
        tensorflow implementation)
        True imply that peepholes are fully reccurent, information flow 
        from any cell states to any gates
        Defaults to False.

    Returns:
        _type_: _description_
    """
    w_init = fax_inits.get(w_init)()
    u_init = fax_inits.get(u_init)()
    #TODO: modify zeros ones init function to respect synthax
    b_init = fax_inits.get(b_init) if b_init in ["ones", "zeros"] else fax_inits.get(b_init)()
    p_init = fax_inits.get(p_init) if p_init in ["ones", "zeros"] else fax_inits.get(p_init)()
    act = fax_funcs.get(activation)
    act_rec = fax_funcs.get(recurrent_activation)
    
    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        """ Initialize the peephole-lstm layer """
        # hidden state init
        states = {}

        cell_states = b_init(rng, (out_dim,), jnp.float32)
        out_states = b_init(rng, (out_dim,), jnp.float32)
        states["cell_states"] = cell_states
        states["out_states"] = out_states
        k1, k2, k3, k4 = random.split(rng, num=4)
        w, u, b, p = (
            w_init(k1, (input_shape[-1], 4*out_dim), jnp.float32),
            u_init(k2, (out_dim, 4*out_dim), jnp.float32),
            b_init(k3, (4*out_dim,), jnp.float32),
            u_init(k4, (out_dim, 3*out_dim), jnp.float32) \
                if full_cell_reccurent else p_init(k4, (3*out_dim,), jnp.float32),
            )
        output_shape = input_shape[:-1] + (out_dim,)\
            if len(input_shape) > 1 else (out_dim,)

        params = {
            "w": w, "u": u, "b": b, "p": p,
            }
        return {}, params, states, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        """ Perform single step update of the network """
        previous_cell_states = states["cell_states"]
        previous_out_states = states["out_states"]
        is_zero = jnp.all(inp == 0.0)
        w, u, b, p = params["w"], params["u"], params["b"], params["p"]
        results = jnp.dot(inp, w)
        i_ff, f_ff, o_ff, c_ff = jnp.split(
            results, 4, axis=-1)
        results = jnp.dot(previous_out_states, u)
        i_fb, f_fb, o_fb, c_fb = jnp.split(results, 4, axis=-1)
        i_b, f_b, o_b, c_b = jnp.split(b, 4, axis=-1)
        i_p, f_p, o_p = jnp.split(p, 3, axis=-1)
        if full_cell_reccurent:
            i_p_c = jnp.dot(previous_cell_states, i_p) 
            f_p_c = jnp.dot(previous_cell_states, f_p)
        else:
            i_p_c = i_p*previous_cell_states
            f_p_c = f_p*previous_cell_states
        i = act_rec(i_ff + i_fb + i_p_c + i_b)
        f = act_rec(f_ff + f_fb + f_p_c + f_b)
        c = act(c_ff + c_fb + c_b)
        c_s = c*i + previous_cell_states*f
        if full_cell_reccurent:
            o_p_c = jnp.dot(c_s, o_p)
        else:
            o_p_c = c_s*o_p
        o = act_rec(o_ff + o_fb + o_p_c + o_b)
        if use_activation_at_output:
            o_s = act(c_s)*o
        else:
            o_s = c_s*o
        states["cell_states"] = lax.select(
            is_zero, previous_cell_states, c_s)
        states["out_states"] = lax.select(
            is_zero, previous_out_states, o_s)
        out_states = (1 - is_zero) * o_s
        return states, out_states

    return init_fun, apply_fun

def lstm(out_dim: int,
         w_init: DispatchFunc = "glorot_normal",
         b_init: DispatchFunc = "normal",
         activation: DispatchFunc = "tanh") -> Layer:
    w_init = fax_inits.get(w_init)()
    b_init = fax_inits.get(b_init) if b_init in ["ones", "zeros"] else fax_inits.get(b_init)()
    act = fax_funcs.get(activation)
    
    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        """ Initialize the GRU layer for stax """
        # hidden state init
        states = {}

        out_states = b_init(rng, (out_dim,), jnp.float32)
        cell_states = b_init(rng, (out_dim,), jnp.float32)

        k1, k2, k3 = random.split(rng, num=3)
        w, u, b = (
            w_init(k1, (input_shape[-1], 4*out_dim), jnp.float32),
            w_init(k2, (out_dim, 4*out_dim), jnp.float32),
            b_init(k3, (4*out_dim,), jnp.float32),
            )
        output_shape = input_shape[:-1] + (out_dim,)\
            if len(input_shape) > 1 else (out_dim,)
        states["out_states"] = out_states
        states["cell_states"] = cell_states
        params = {
            "w": w, "u": u, "b": b
            }
        return {}, params, states, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        """ Perform single step update of the network """
        previous_out_states = states["out_states"]
        previous_cell_states = states["cell_states"]
        is_zero = jnp.all(inp == 0.0)
        w, u, b = params["w"], params["u"], params["b"]
        results = jnp.dot(inp, w)
        i_ff, f_ff, o_ff, c_ff = jnp.split(results, 4, axis=-1)
        results = jnp.dot(previous_out_states, u)
        i_fb, f_fb, o_fb, c_fb = jnp.split(results, 4, axis=-1)
        i_b, f_b, o_b, c_b = jnp.split(b, 4, axis=-1)
        i = nn.sigmoid(i_ff + i_fb + i_b)
        f = nn.sigmoid(f_ff + f_fb + f_b)
        o = nn.sigmoid(o_ff + o_fb + o_b)
        c = nn.tanh(c_ff + c_fb + c_b)
        c_s = f * previous_cell_states + i * c
        o_s = o * act(c_s)
        # states["out_states"] = lax.select(
        #     is_zero, previous_out_states, o_s)
        # states["cell_states"] = lax.select(
        #     is_zero, previous_cell_states, c_s)
        states["out_states"] = o_s
        states["cell_states"] = c_s
        # o_s = (1 - is_zero) * o_s
        return states, o_s

    return init_fun, apply_fun

def gru(out_dim: int,
        w_init: DispatchFunc = "glorot_normal",
        b_init: DispatchFunc = "normal",
        activation: DispatchFunc = "tanh") -> Layer:
    w_init = fax_inits.get(w_init)()
    b_init = fax_inits.get(b_init) if b_init in ["ones", "zeros"] else fax_inits.get(b_init)()
    act = fax_funcs.get(activation)

    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        """ Initialize the GRU layer for stax """
        # hidden state init
        states = {}

        hidden = b_init(rng, (out_dim,), jnp.float32)

        k1, k2, k3, k4 = random.split(rng, num=4)
        w, u, u_out, b = (
            w_init(k1, (input_shape[-1], 3*out_dim), jnp.float32),
            w_init(k2, (out_dim, 2*out_dim), jnp.float32),
            w_init(k3, (out_dim, out_dim), jnp.float32),
            b_init(k4, (3*out_dim,), jnp.float32),
            )
        output_shape = input_shape[:-1] + (out_dim,)\
            if len(input_shape) > 1 else (out_dim,)
        states["hidden"] = hidden
        params = {
            "w": w, "u": u, "u_out": u_out, "b": b
            }
        return {}, params, states, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        """ Perform single step update of the network """
        previous_hidden = states["hidden"]
        is_zero = jnp.all(inp == 0.0)
        w, u, u_out, b = params["w"], params["u"], params["u_out"], params["b"]
        results = jnp.dot(inp, w)
        k_update_ff, k_reset_ff, k_out_ff = jnp.split(results, 3, axis=-1)
        results = jnp.dot(previous_hidden, u)
        k_update_fb, k_reset_fb = jnp.split(results, 2, axis=-1)
        update_b, reset_b, out_b = jnp.split(b, 3, axis=-1)
        update_gate = nn.sigmoid(k_update_ff + k_update_fb + update_b)
        reset_gate = nn.sigmoid(k_reset_ff + k_reset_fb + reset_b)
        output_gate = act(k_out_ff
                              + jnp.dot(jnp.multiply(
                                  reset_gate, previous_hidden), u_out)
                              + out_b)
        output = jnp.multiply(update_gate, previous_hidden) +\
            jnp.multiply(1 - update_gate, output_gate)
        states["hidden"] = output
        # output = (1 - is_zero) * output
        return states, output

    return init_fun, apply_fun


def oja_rule(inverted: bool, h_max: float, gamma_pos: float, gamma_neg: float,
             branch_normalization: bool = False) -> LocalLayer:
    def init_fun(rng: PRNGKey, init_params: dict, 
                 init_states: dict, input_shape):
        if isinstance(gamma_pos, Sequence):
            gamma_pos_param = jnp.logspace(gamma_pos[0], gamma_pos[1], 
                                           input_shape[0])
        else:
            gamma_pos_param = nn.initializers.ones(rng, input_shape) * gamma_pos
        if isinstance(gamma_neg, Sequence):
            gamma_neg_param = jnp.logspace(gamma_neg[0], gamma_neg[1], 
                                           input_shape[0])
        else:
            gamma_neg_param = nn.initializers.ones(rng, input_shape) * gamma_neg
        h_max_param = nn.initializers.ones(rng, input_shape) * h_max
        params = {
            "h_max": h_max_param,
            "gamma_pos": gamma_pos_param,
            "gamma_neg": gamma_neg_param,
        }
        return params, input_shape

    def apply_fun(params: dict, h_previous: Array, k: Array, v: Array):
        h_max, gamma_pos, gamma_neg = params["h_max"], params["gamma_pos"],\
            params["gamma_neg"]
        if len(jnp.shape(h_previous)) == 2 \
                and h_previous.shape[0] == h_previous.shape[1]:
            k = jnp.expand_dims(k, 1)
            v = jnp.expand_dims(v, 0)
        if inverted:
            inib = k ** 2
        else:
            inib = v ** 2
        k_v = k * v
        if branch_normalization:
            kv_sum = jnp.sum(k_v, axis=0)
            kv_normalized = (kv_sum - k_v) / (k.shape[0] - 1)
            h_max = h_max * jax.lax.stop_gradient(jnp.exp(-kv_normalized))

        return nn.relu(h_previous + gamma_pos * (
            h_max - h_previous) * k_v - gamma_neg * h_previous * inib)

    return init_fun, apply_fun


def hebbian_rule(h_max: float, gamma_pos: Union[float, tuple[float, float]],
                 gamma_neg: Union[float, tuple[float, float]]) -> LocalLayer:
    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        h_max_param = nn.initializers.ones(rng, input_shape) * h_max
        if isinstance(gamma_pos, Sequence):
            gamma_pos_param = jnp.logspace(gamma_pos[0], gamma_pos[1], 
                                           input_shape[0])
        else:
            gamma_pos_param = nn.initializers.ones(rng, input_shape) * gamma_pos
        if isinstance(gamma_neg, Sequence):
            gamma_neg_param = jnp.logspace(gamma_neg[0], gamma_neg[1], 
                                           input_shape[0])
        else:
            gamma_neg_param = nn.initializers.ones(rng, input_shape) * gamma_neg
        params = {
            "h_max": h_max_param,
            "gamma_pos": gamma_pos_param,
            "gamma_neg": gamma_neg_param
        }
        return params, input_shape

    def apply_func(params, h_previous, k, v) -> Array:
        h_max, gamma_pos, gamma_neg = params["h_max"], params["gamma_pos"],\
            params["gamma_neg"]
        if len(jnp.shape(h_previous)) == 2 \
                and h_previous.shape[0] == h_previous.shape[1]:
            k = jnp.expand_dims(k, 1)
            v = jnp.expand_dims(v, 0)
        return nn.relu(h_previous + gamma_pos * (
            h_max - h_previous) * k * v - gamma_neg * h_previous)

    return init_fun, apply_func


def h_mem_local_rule(hebbian_type: str, hebbian_params: dict) -> tuple:
    hebbian_type = hebbian_type.lower()
    if hebbian_type == "oja" or hebbian_type == "reversed_oja":
        reverse_flag = hebbian_type == "reversed_oja"
        local_rule = oja_rule(reverse_flag, **hebbian_params)
    elif hebbian_type == "hebbian":
        local_rule = hebbian_rule(**hebbian_params)
    else:
        raise NotImplementedError
    return local_rule


def branched_hmem(memory_size: int, n_branch: int, scaling_by_branch: bool,
                  hebbian_type: str, hebbian_params: dict,
                  store_func: DispatchFunc = "relu",
                  recall_func: DispatchFunc = "relu",
                  w_init: DispatchFunc = "orthogonal",
                  b_init: DispatchFunc = "zeros",
                  h_init: DispatchFunc = "zeros") -> Layer:
    w_init = fax_inits.get(w_init)()
    b_init = fax_inits.get(b_init)
    h_init = fax_inits.get(h_init)
    memory_shape = (n_branch, memory_size)
    init_local_rule, apply_local_rule = h_mem_local_rule(
        hebbian_type, hebbian_params)
    store_func = fax_funcs.get(store_func)
    recall_func = fax_funcs.get(recall_func)

    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape, *args):
        """ Initialize the h_mem layer for seq_stax """
        # hidden state init
        hidden = h_init(rng, memory_shape)
        states = {}
        out_dim = memory_shape[-1]

        keys, *subkeys = random.split(rng, num=7)
        key_kernel, value_kernel, key_bias, value_bias = (
            w_init(subkeys[0], (memory_shape[0], input_shape[-1],
                                memory_shape[1]), jnp.float32),
            w_init(subkeys[1], (input_shape[-1], out_dim), jnp.float32),
            b_init(subkeys[3], memory_shape, jnp.float32),
            b_init(subkeys[4], (out_dim,), jnp.float32),
        )
        output_shape = input_shape[:-1] + (out_dim,)\
            if len(input_shape) > 1 else (out_dim,)
        states["hidden"] = hidden
        hebbian_params, _ = init_local_rule(rng, None, None, (n_branch, 1))
        params = {"key_kernel": key_kernel, "value_kernel": value_kernel,
                  "key_bias": key_bias, "value_bias": value_bias,
                  "hebbian_params": hebbian_params}

        return {}, params, states, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp, fact_type):
        is_zero = jnp.all(inp == 0.0)
        previous_hidden = states["hidden"]
        key_kernel, value_kernel = params["key_kernel"], params["value_kernel"]
        key_bias, value_bias = params["key_bias"], params["value_bias"]
        hebbian_params = params["hebbian_params"]

        new_hidden = lax.select(
            fact_type == ObsType.store,
            apply_local_rule(
                hebbian_params,
                previous_hidden,
                store_func(jnp.dot(inp, key_kernel) + key_bias),
                store_func(jnp.dot(inp, value_kernel) + value_bias)),
            previous_hidden)

        states["hidden"] = lax.select(is_zero, previous_hidden, new_hidden)
        output = recall_func(jnp.dot(inp, key_kernel) + key_bias) * new_hidden
        if scaling_by_branch:
            branches_weights = lax.stop_gradient(
                nn.softmax(new_hidden, axis=0))
            output = branches_weights * output
        output = jnp.sum(output, axis=0)
        return states, (1 - is_zero) * output
    return init_fun, apply_fun


def recurrent_hmem(memory_size: int, hebbian_type: str, hebbian_params: dict,
                   modulation: str = "additive",
                   modulation_func: DispatchFunc = "relu",
                   store_func: DispatchFunc = "relu",
                   recall_func: DispatchFunc = "relu",
                   to_modulate: str = "output",
                   w_init: DispatchFunc = "orthogonal",
                   u_init: DispatchFunc = "orthogonal",
                   b_init: DispatchFunc = "zeros",
                   h_init: DispatchFunc = "zeros"):
    w_init = fax_inits.get(w_init)()
    u_init = fax_inits.get(u_init)()
    b_init = fax_inits.get(b_init)
    h_init = fax_inits.get(h_init)

    memory_shape = (memory_size,)
    init_local_rule, apply_local_rule = h_mem_local_rule(hebbian_type,
                                                         hebbian_params)
    store_func = fax_funcs.get(store_func)
    recall_func = fax_funcs.get(recall_func)
    modulation_func = fax_funcs.get(modulation_func)

    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape, *args):
        """ Initialize the h_mem layer for seq_stax """
        # hidden state init
        keys, *subkeys = random.split(rng, num=9)
        hidden = h_init(subkeys[0], memory_shape)
        prev_z = h_init(subkeys[1], memory_shape)
        states = {}
        out_dim = memory_shape[0]

        key_kernel, value_kernel, u_kernel, key_bias, value_bias, u_bias = (
            w_init(subkeys[2], (input_shape[-1], out_dim), jnp.float32),
            w_init(subkeys[3], (input_shape[-1], out_dim), jnp.float32),
            u_init(subkeys[4], (out_dim, out_dim), jnp.float32),
            b_init(subkeys[5], (out_dim,), jnp.float32),
            b_init(subkeys[6], (out_dim,), jnp.float32),
            b_init(subkeys[7], (out_dim,), jnp.float32),

        )
        output_shape = input_shape[:-1] + (out_dim,)\
            if len(input_shape) > 1 else (out_dim,)
        states["hidden"] = hidden
        states["z"] = prev_z

        hebbian_params, _ = init_local_rule(rng, None, None, ())
        params = {"key_kernel": key_kernel, "value_kernel": value_kernel,
                  "u_kernel": u_kernel, "u_bias": u_bias,
                  "key_bias": key_bias, "value_bias": value_bias,
                  "hebbian_params": hebbian_params}

        return {}, params, states, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp, fact_type):
        is_zero = jnp.all(inp == 0.0)
        previous_hidden = states["hidden"]
        previous_z = jax.lax.stop_gradient(states["z"])

        key_kernel, value_kernel, u_kernel = params["key_kernel"],\
            params["value_kernel"], params["u_kernel"]
        key_bias, value_bias, u_bias = params["key_bias"],\
            params["value_bias"], params["u_bias"]
        hebbian_params = params["hebbian_params"]
        k = recall_func(jnp.dot(inp, key_kernel) + key_bias)
        rec_z = modulation_func(jnp.dot(previous_z, u_kernel) + u_bias)
        v = store_func(jnp.dot(inp, value_kernel) + value_bias)
        if modulation == "multiplicative":
            new_v = v * rec_z
        else:
            new_v = v + rec_z
        new_hidden = lax.select(
            fact_type == ObsType.store,
            apply_local_rule(
                hebbian_params,
                previous_hidden,
                store_func(jnp.dot(inp, key_kernel) + key_bias), new_v),
            previous_hidden)

        states["hidden"] = lax.select(is_zero, previous_hidden, new_hidden)
        output = k * new_hidden
        flag = fact_type == ObsType.recall
        if to_modulate == "value":
            states["z"] = lax.select(flag, previous_z, v)
        elif to_modulate == "key":
            states["z"] = lax.select(flag, previous_z, k)
        elif to_modulate == "output":
            states["z"] = lax.select(flag, previous_z, output)
        else:
            raise NotImplementedError(
                f"Modulation of {to_modulate} is not implemented")

        return states, (1 - is_zero) * output

    return init_fun, apply_fun


def hmem_sr(memory_size: int, hebbian_type: str, hebbian_params: dict,
            store_func: DispatchFunc = "relu",
            recall_func: DispatchFunc = "relu",
            w_init: DispatchFunc = "orthogonal",
            b_init: DispatchFunc = "zeros",
            h_init: DispatchFunc = "zeros"):
    if w_init == "orthogonal":
        scale = jnp.sqrt(2) if store_func == "relu" else 1
        w_init = fax_inits.get(w_init)(scale=scale)
    else:
        w_init = fax_inits.get(w_init)()
    b_init = fax_inits.get(b_init)
    h_init = fax_inits.get(h_init)
    store_func = fax_funcs.get(store_func)
    recall_func = fax_funcs.get(recall_func)
    memory_shape = (memory_size,)
    init_local_rule, apply_local_rule = h_mem_local_rule(hebbian_type,
                                                         hebbian_params)
    def init_fun(rng: PRNGKey, init_params: dict, 
                 init_states: dict, input_shape, *args):
        """ Initialize the h_mem successor representation like """
        hidden = h_init(rng, memory_shape)
        prev_input = h_init(rng, memory_shape)
        states = {}
        out_dim = memory_shape[0]

        keys, *subkeys = random.split(rng, num=7)
        key_kernel, value_kernel, key_bias, value_bias = (
            w_init(subkeys[0], (input_shape[-1], out_dim), jnp.float32),
            w_init(subkeys[1], (input_shape[-1], out_dim), jnp.float32),
            b_init(subkeys[3], (out_dim,), jnp.float32),
            b_init(subkeys[4], (out_dim,), jnp.float32),
        )
        output_shape = input_shape[:-1] + (out_dim,)\
            if len(input_shape) > 1 else (out_dim,)
        states["hidden"] = hidden
        states["prev_input"] = prev_input
        
        hebbian_params, _ = init_local_rule(rng, None, None, ())
        params = {"key_kernel": key_kernel, "value_kernel": value_kernel,
                  "key_bias": key_bias, "value_bias": value_bias,
                  "hebbian_params": hebbian_params}

        return {}, params, states, output_shape
    
    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp, fact_type):

        is_zero = jnp.all(inp == 0.0)
        previous_hidden = states["hidden"]
        previous_input = states["prev_input"]
        key_kernel, value_kernel = params["key_kernel"], params["value_kernel"]
        key_bias, value_bias = params["key_bias"], params["value_bias"]
        hebbian_params = params["hebbian_params"]
        new_hidden = jax.lax.select(
            fact_type == ObsType.recall,
            previous_hidden,
            apply_local_rule(
                hebbian_params, previous_hidden,
                recall_func(jnp.dot(previous_input, key_kernel) + key_bias),
                recall_func(jnp.dot(inp, value_kernel) + value_bias))
            )

        k = recall_func(jnp.dot(inp, key_kernel) + key_bias)
        states["hidden"] = lax.select(is_zero, previous_hidden, new_hidden)
        states["prev_input"] = lax.select(is_zero, previous_input, inp)
        output = k * new_hidden
        return states, (1 - is_zero) * output
    return init_fun, apply_fun
def pyramidal_block(nb_nodes:int, nb_branches_per_node:int, nb_synapse_per_branchs: int, activation):
    pass
def pyramidal_node(nb_branches:int, nb_synapses:int, activation: DispatchFunc = "relu"):
    pass
    """
    white paper:
    pyramidal node receive a up-steam u_i value for each branch i
    the branch produce an activity k_i = relu(W_i input + u_i)
    alternative: 
        add a weight for k_i in W 
    W is learn by GHL w.r.t the other branches.
    with then compute 
        z <-  sum(h_i k_i) where h_i is the branch 
    potentiation stength of i
    z is transmetted to down-stream path
    branches received a down-stream signal v 
    d_h_i/d_t <- (1 - h_i)*k_i*v - h_i*k_i**2
    branch i transmit to up-stream the signal:
    v_new <- h_i*v + k_i
    h_i musn't grow to large and preferentially need to be btw (0, 1)
    alternative:
        maybe sigmoid is added after updating h_i)
    
    flow of data:
        u_i is given by feedforward signal
        v is a backpropagated signal
        z go to down-stream (output)
        d_W is learn during backprop
        alternate:
            directly update W during feedforward
            v is a feed-back signal
            store v_new in state in order to transmit it as feed-back to the
            up-steamn
        u_i need to be store in state
        inputs come either from inputs neurons of from the output of the neurons layer
        from which branch are part of
    
    Args:
        nb_branches (int): _description_
        nb_synapses (int): _description_
        activation (DispatchFunc, optional): _description_. Defaults to "relu".
    """
def hmem(memory_size: int, hebbian_type: str, hebbian_params: dict,
         store_func: DispatchFunc = "relu",
         recall_func: DispatchFunc = "relu",
         is_eprop: bool = False,
         is_diagonal: bool = False,
         is_fact_agnostic: bool = False,
         is_reccurrent: bool = False,
         w_init: DispatchFunc = "orthogonal",
         b_init: DispatchFunc = "zeros",
         h_init: DispatchFunc = "zeros",
         ) -> Layer:
    if w_init == "orthogonal":
        scale = jnp.sqrt(2) if store_func == "relu" else 1
        w_init = fax_inits.get(w_init)(scale=scale)
    else:
        w_init = fax_inits.get(w_init)()
        
    # local_derivation define all the internal computations
    # for the new hidden state, this allow to extract the vjp in terms
    # of the hidden state more easily at the price of computing k twice
    def local_derivation(key_kernel, value_kernel, key_bias, value_bias,
                         hebb_params, previous_hidden, value, fact_type):
        if is_reccurrent:
            inp, _ = jnp.split(value, [len(value) - memory_size,])
            v = store_func(jnp.dot(inp, value_kernel) + value_bias)
        else:
            v = store_func(jnp.dot(value, value_kernel) + value_bias)
        k = store_func(jnp.dot(value, key_kernel) + key_bias)
        if is_fact_agnostic:
            new_hidden = apply_local_rule(hebb_params, previous_hidden, k, v)
        else:
            new_hidden = lax.select(
                fact_type == ObsType.store,
                apply_local_rule(hebb_params, previous_hidden, k, v),
                previous_hidden)
        return new_hidden
    b_init = fax_inits.get(b_init)
    h_init = fax_inits.get(h_init)
    store_func = fax_funcs.get(store_func)
    recall_func = fax_funcs.get(recall_func)
    memory_shape = (memory_size,) \
        if is_diagonal else (memory_size, memory_size)
    init_local_rule, apply_local_rule = h_mem_local_rule(hebbian_type,
                                                         hebbian_params)
    
    def init_fun(rng: PRNGKey, init_params: dict, 
                 init_states: dict, input_shape, *args):
        """ Initialize the h_mem layer """
        # hidden state init
        hidden = h_init(rng, memory_shape)
        params = {}
        states = {}
        out_dim = memory_shape[0]

        states["hidden"] = hidden
        keys, *subkeys = random.split(rng, num=7)
        _input_shape = input_shape[-1]
        if is_reccurrent:
            _input_shape += out_dim
            states["z"] = hidden
        key_kernel, value_kernel, key_bias, value_bias = (
            w_init(subkeys[0], (_input_shape, out_dim), jnp.float32),
            w_init(subkeys[1], (input_shape[-1], out_dim), jnp.float32),
            b_init(subkeys[3], (out_dim,), jnp.float32),
            b_init(subkeys[4], (out_dim,), jnp.float32),
        )
        
        output_shape = input_shape[:-1] + (out_dim,)\
            if len(input_shape) > 1 else (out_dim,)
            
        if is_eprop:
            states["key_kernel_trace"] = jnp.zeros_like(key_kernel)
            states["value_kernel_trace"] = jnp.zeros_like(value_kernel)
            states["key_bias_trace"] = jnp.zeros_like(key_bias)
            states["value_bias_trace"] = jnp.zeros_like(value_bias)

        hebbian_params, _ = init_local_rule(rng, None, None, (memory_size, ))
        params |= {"key_kernel": key_kernel, "value_kernel": value_kernel,
                  "key_bias": key_bias, "value_bias": value_bias,
                  "hebbian_params": hebbian_params}
        states["value"] = hidden
        states["key"] = hidden

        return {}, params, states, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp, fact_type):

        is_zero = jnp.all(inp == 0.0)
        previous_hidden = states["hidden"]
        key_kernel, value_kernel = params["key_kernel"], params["value_kernel"]
        key_bias, value_bias = params["key_bias"], params["value_bias"]
        hebbian_params = params["hebbian_params"]
        v = recall_func(jnp.dot(inp, value_kernel) + value_bias)
        if is_reccurrent:
            z = states["z"]
            inp_p_zero = jnp.concatenate((inp, jnp.zeros_like(z)), axis=-1)
            inp = jnp.concatenate((inp, z), axis=-1)
            k = recall_func(jnp.dot(inp_p_zero, key_kernel) + key_bias)
        else:
            k = recall_func(jnp.dot(inp, key_kernel) + key_bias)
        new_hidden = local_derivation(key_kernel, value_kernel, 
                                      key_bias, value_bias,
                                      hebbian_params, previous_hidden, 
                                      inp, fact_type)

        states["hidden"] = lax.select(is_zero, previous_hidden, new_hidden)
        states["value"] = (1 - is_zero) * v
        states["key"] = (1 - is_zero) * k


        if len(jnp.shape(new_hidden)) == 2:
            output = jnp.dot(k, new_hidden)
        else:
            output = k * new_hidden
        #TODO: temporary export key and value for logging, remove after usage

        if is_reccurrent:
            states["z"] = (1 - is_zero) * new_hidden + is_zero * z
        return states, (1 - is_zero) * output

    def apply_fun_eprop(rng: PRNGKey, params: dict, 
                        states: dict, inp, fact_type):
        states = jax.lax.stop_gradient(states)
        inp = jax.lax.stop_gradient(inp)
        is_zero = jnp.all(inp == 0.0)
        key_kernel = params["key_kernel"]
        key_bias = params["key_bias"]
        value_kernel = params["value_kernel"]
        value_bias = params["value_bias"]
        v = recall_func(jnp.dot(inp, value_kernel) + value_bias)
        if is_reccurrent:
            z = states["z"]
            inp = jnp.concatenate((inp, z), axis=-1)
            # states, output = apply_drop(rng, {}, states, output)
        k = recall_func(jnp.dot(inp, key_kernel) + key_bias)

        new_hidden, states = apply_vjp_fun_eprop(rng, params, states, inp,
                                                 fact_type)
        states = jax.lax.stop_gradient(states)
        states["value"] = (1 - is_zero) * v
        states["key"] = (1 - is_zero) * k
        output = k * new_hidden
        if is_reccurrent:
            states["z"] = jax.lax.stop_gradient((1 - is_zero) * output + is_zero * z)
        return states, (1 - is_zero) * output

    @jax.custom_vjp
    def apply_vjp_fun_eprop(rng: PRNGKey, params: dict,
                            states: dict, inp, fact_type):
        """ Perform single step update of the network """
        is_zero = jnp.all(inp == 0.0)
        key_kernel, value_kernel = params["key_kernel"], params["value_kernel"]
        key_bias, value_bias = params["key_bias"], params["value_bias"]
        hebbian_params = params["hebbian_params"]

        previous_hidden = states["hidden"]
        new_hidden, vjp_hidden = jax.vjp(lambda w_k, w_v, b_k, b_v, _h:
                                         local_derivation(w_k, w_v,
                                                          b_k, b_v,
                                                          hebbian_params,
                                                          _h, inp, fact_type),
                                         key_kernel, value_kernel,
                                         key_bias, value_bias, previous_hidden)
        D_h_D_w_k = states["key_kernel_trace"]
        D_h_D_w_v = states["value_kernel_trace"]
        D_h_D_b_k = states["key_bias_trace"]
        D_h_D_b_v = states["value_bias_trace"]

        d_h_d_w_k, d_h_d_w_v, d_h_d_b_k, d_h_d_b_v, d_h_d_h = vjp_hidden(
            jnp.ones_like(new_hidden))

        states["key_kernel_trace"] = lax.select(
            is_zero, D_h_D_w_k, d_h_d_h * D_h_D_w_k + d_h_d_w_k)
        states["value_kernel_trace"] = lax.select(
            is_zero, D_h_D_w_v, d_h_d_h * D_h_D_w_v + d_h_d_w_v)
        states["key_bias_trace"] = lax.select(
            is_zero, D_h_D_b_k, d_h_d_h * D_h_D_b_k + d_h_d_b_k)
        states["value_bias_trace"] = lax.select(
            is_zero, D_h_D_b_v, d_h_d_h * D_h_D_b_v + d_h_d_b_v)

        new_hidden = lax.select(is_zero, previous_hidden, new_hidden)
        states["hidden"] = jax.lax.stop_gradient(new_hidden)

        return new_hidden, states

    def f_fwd(rng: PRNGKey, params: dict, states: dict, inp, fact_type):
        new_hidden, new_states = apply_vjp_fun_eprop(rng, params, 
                                                     states, inp, fact_type)
        hebbian_params = tree_map(
            lambda x: jnp.zeros_like(x), params["hebbian_params"])

        return (new_hidden, new_states), (new_states, hebbian_params)

    def f_bwd(res, g):
        z_dot = g[0]
        states, hebbian_params = res
        D_h_D_w_k = states["key_kernel_trace"]
        D_h_D_w_v = states["value_kernel_trace"]
        D_h_D_b_k = states["key_bias_trace"]
        D_h_D_b_v = states["value_bias_trace"]

        return None, {"key_kernel": z_dot * D_h_D_w_k,
                      "value_kernel": z_dot * D_h_D_w_v,
                      "key_bias": z_dot * D_h_D_b_k,
                      "value_bias": z_dot * D_h_D_b_v,
                      "hebbian_params": hebbian_params}, None, None, None

    apply_vjp_fun_eprop.defvjp(f_fwd, f_bwd)
    apply_fun = apply_fun_eprop if is_eprop else apply_fun

    return init_fun, apply_fun

def word_embedding(out_dim: int, vocab_size: int,
                   w_init: DispatchFunc = "one_hot") -> Layer:
    w_init_str = w_init
    w_init = fax_inits.get(w_init)
    if w_init_str != "one_hot":
        w_init = w_init()

    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        w = w_init(rng, (vocab_size, out_dim), jnp.float32)
        output_shape = input_shape + (out_dim,)
        params = {"w": w}
        return {}, params, {}, output_shape

    def apply_fun(rng: PRNGKey, apply_params: dict,
                  apply_states: dict, inp) -> tuple[dict, dict]:
        w = apply_params["w"]
        word_idx = inp.astype(jnp.int32)
        output = w[word_idx]
        zeros = jnp.where(inp == 0.0, inp, 1.0)
        output = output * jnp.expand_dims(zeros, 1)
        return {}, output

    return init_fun, apply_fun


def phrase_embedding(type: str = "bow",
                     w_init: DispatchFunc = "glorot_normal"):
    w_init_str = w_init
    w_init = fax_inits.get(w_init)
    if w_init_str != "one_hot":
        w_init = w_init()
    type = type.lower()

    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        params = {}
        output_shape = input_shape[-1]
        if type == "position":
            " https://arxiv.org/pdf/1503.08895.pdf"
            p_emb = np.zeros(input_shape, dtype=float)
            ls = input_shape[0] + 1
            le = input_shape[1] + 1
            for i in range(1, ls):
                for j in range(1, le):
                    p_emb[i - 1, j - 1] = (i - (input_shape[1] + 1) / 2) * \
                        (j - (input_shape[0] + 1) / 2)
            p_emb = 1 + 4 * p_emb / input_shape[1] / input_shape[0]
            pos_emb = jnp.array(p_emb, dtype=jnp.float32)
            params["w"] = pos_emb
        else:
            w = w_init(rng, input_shape)
            params["w"] = w
        return {}, params, {}, (output_shape,)

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        w = params["w"]
        output = (w * inp)
        output = jnp.sum(output, axis=-2)
        return {}, output

    return init_fun, apply_fun


def layer_normalization(normalization_type: str,
                        epsi: float = 1e-5) -> Layer:
    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        return {}, {}, {}, input_shape

    def apply_func_min_max(rng: PRNGKey, params: dict, states: dict, inp):
        _max = jnp.max(inp, axis=-1)
        _min = jnp.min(inp, axis=-1)
        scaled_output = (inp - _min) / (_max - _min + epsi)
        return {}, scaled_output

    def apply_func_mean_var(rng: PRNGKey, params: dict, states: dict, inp):
        output_mean = jnp.mean(inp, axis=-1)
        output_var = jnp.var(inp, axis=-1)
        scaled_output = (inp - output_mean) * lax.rsqrt(output_var + epsi)
        return {}, scaled_output

    if normalization_type == "min_max":
        apply_fun = apply_func_min_max
    elif normalization_type == "mean_var":
        apply_fun = apply_func_mean_var
    else:
        raise NotImplementedError(
            f"Layer normalization {normalization_type} is not implemented")

    return init_fun, apply_fun


def temporal_normalization(gamma_init: DispatchFunc = "ones",
                           beta_init: DispatchFunc = "zeros",
                           epsi: float = 1e-5) -> Layer:
    """
    Perform an online z-score normalization over time
    :return:
    """
    gamma_init = fax_inits.get(gamma_init)
    beta_init = fax_inits.get(beta_init)
    

    def init_fun(rng: PRNGKey, init_params: dict,
                 init_states: dict, input_shape):
        output_shape = input_shape
        states = {"mean": nn.initializers.zeros(rng, (input_shape[-1],)),
                  "M": nn.initializers.zeros(rng, (input_shape[-1],)),
                  "n": nn.initializers.ones(rng, ())}

        # M_{2,t} = M_{2,t-1} + (x_t - mean_{t-1}(x_t - mean_t)
        beta = beta_init(rng, output_shape, jnp.float32)
        gamma = gamma_init(rng, output_shape, jnp.float32)
        params = {"beta": beta, "gamma": gamma}

        return {}, params, states, output_shape

    def apply_fun(rng: PRNGKey, params: dict, states: dict, inp):
        n = states["n"]
        m = states["M"]
        flag = n >= 2.0
        previous_mean = states["mean"]
        is_zero = jnp.all(inp == 0.0)
        beta, gamma = params["beta"], params["gamma"]
        # stable online mean variance reduction
        new_mean = lax.stop_gradient(previous_mean + (inp - previous_mean) / n)
        new_m = lax.stop_gradient(m + (inp - previous_mean) * (inp - new_mean))
        new_var = lax.select(flag, new_m / (n - 1.0), new_m)
        new_x = lax.select(flag, beta + ((inp - new_mean) * lax.rsqrt(new_var + epsi)) * gamma, inp)

        states["n"] = lax.select(is_zero, n, n + 1.0)
        states["M"] = lax.select(is_zero, m, new_m)
        states["mean"] = lax.select(is_zero, previous_mean, new_mean)
        new_x = lax.select(is_zero, inp, new_x)
        return states, new_x

    return init_fun, apply_fun


_layers_dispatch = {
    "dropout": dropout,
    "projection": projection,
    "temporal_normalization": temporal_normalization,
    "layer_normalization": layer_normalization,
    "phrase_embedding": phrase_embedding,
    "word_embedding": word_embedding,
    "hmem": hmem,
    "hmem_sr": hmem_sr,
    "rec_hmem": recurrent_hmem,
    "branched_hmem": branched_hmem,
    "gru": gru,
    "lstm": lstm,
    "peephole_lstm": peephole_lstm,
    "dense": dense,
    "identity": identity,
    "concat": concat,
    "sum": summation,
    "gaussian": gaussian,
    "dirichlet": dirichlet,
    "conv_0D": conv_0D,
    "conv_1D": conv_1D,
    "conv_2D": conv_2D,
    "conv_3D": conv_3D,
    "batch_norm": batch_norm,
    "max_pool": max_pool,
    "sum_pool": sum_pool,
    "avg_pool": avg_pool,
    "flatten": flatten,
    "application": application,
    "frozen_input": frozen_input,
    "empty": empty,
    
}


def get(identifier: Union[str, Callable]):
    if isinstance(identifier, str):
        try:
            return _layers_dispatch[identifier]
        except KeyError:
            valid_ids_msg = "\n".join(_layers_dispatch.keys())
            print(f"{identifier} does not exist in the lookup table \n"
                  f"valid identifier are:\n {valid_ids_msg}")
    elif isinstance(identifier, Callable):
        return identifier
