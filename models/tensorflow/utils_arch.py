import numpy as np
import tensorflow as tf
import six
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.distribute import distribution_strategy_context
from functools import partial

def euclidean_distance(x, use_fp16=False, use_sqrt=False):
    #x: batchsize*node*featdim  => batchsize*node*node
    dot_product =  tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
    square_norm = tf.linalg.diag_part(dot_product)

    distances = tf.expand_dims(square_norm, 2) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances = tf.maximum(distances, 0.0)

    mask = tf.cast(tf.equal(distances, 0.0), tf.float32 if not use_fp16 else tf.float16)
    distances = distances + mask * (1e-16 if not use_fp16 else 1e-5)

    if use_sqrt:
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances

def norm_lap(A):
    D = tf.reduce_sum(A, axis=2)
    inv_sqrt = tf.pow(D, -0.5)
    inv_sqrt = tf.where(tf.math.is_inf(inv_sqrt), tf.zeros_like(inv_sqrt), inv_sqrt)
    inv_sqrt = tf.linalg.diag(inv_sqrt)

    return tf.matmul(tf.matmul(inv_sqrt, tf.linalg.diag(D)- A), tf.transpose(inv_sqrt, [0, 2, 1]))

def adaptive_sigma(dx, y, eps=1e-3, use_fp16=False):
    T = tf.cast(tf.equal(tf.expand_dims(y,1), tf.expand_dims(y, 2)), tf.float32 if not use_fp16 else tf.float16)
    F =1.0-T
    P = tf.multiply(T, dx)
    N = tf.multiply(F, dx)

    a = tf.reduce_sum(P, axis=[1,2], keepdims=True)/tf.reduce_sum(T, axis=[1,2], keepdims=True)
    b = tf.reduce_sum(N, axis=[1,2], keepdims=True)/tf.reduce_sum(F, axis=[1,2], keepdims=True)

    a2 = tf.math.square(a)
    b2 = tf.math.square(b+eps)

    sigma = tf.math.sqrt((a2-b2)/(2.0*tf.math.log(a2/b2)))

    return sigma

def knn_edge(dx, nodenum, gamma, eps=1e-7, use_fp16=False):
    batchsize = tf.shape(dx)[0]
    
    #remove self
    dx = tf.linalg.set_diag(dx, (1e10 if not use_fp16 else 1e5)*tf.ones([batchsize, nodenum], dtype=tf.float32 if not use_fp16 else tf.float16))

    #allgraph: labeled nodes vs labeled nodes
    _, indices = tf.nn.top_k(-dx, k=gamma, sorted=False)

    bindices = tf.tile(tf.expand_dims(tf.range(0, batchsize)* nodenum * nodenum, 1), (1, nodenum*gamma))
    gindices = tf.reshape(tf.tile(tf.expand_dims(tf.range(0, nodenum) * nodenum, 1),(batchsize, gamma)), [batchsize,-1])
    gindices = gindices + tf.reshape(indices, [batchsize, -1]) + bindices
    gindices = tf.reshape(gindices,(-1,1))
    n_mask  = tf.scatter_nd(gindices, tf.ones(tf.shape(gindices),  tf.float32 if not use_fp16 else tf.float16), tf.shape(tf.reshape(dx,[-1,1])))
    n_mask = tf.reshape(n_mask, tf.shape(dx))

    n_mask = tf.maximum(n_mask, tf.transpose(n_mask, [0,2,1]))

    return n_mask

def binarize(y, c):
    return tf.where(tf.equal(y, c), tf.ones_like(y, tf.float32), tf.negative(tf.ones_like(y, tf.float32)))

def binarize_gs(y):
    return tf.multiply(tf.where(tf.equal(y, 1), tf.ones_like(y, tf.float32), tf.negative(tf.ones_like(y, tf.float32))), tf.cast(tf.not_equal(y, 0), tf.float32))

def edge_weighting(dx, sigma=None, s=None, e=None, eps=1e-3, directed=False, use_fp16=False):
    if s is not None:
        ms = tf.sqrt(eps + tf.multiply(tf.expand_dims(s, 2), tf.expand_dims(s, 1)))
        dx = tf.multiply(dx, ms)

    if sigma is not None:
        if sigma is tf.Tensor:
            w = tf.exp(-tf.square(dx)/(2.0*tf.square(sigma)))
        else:
            w = tf.exp(-tf.square(dx)/(2.0*(sigma**2)))
    else:
        w = tf.exp(-tf.square(dx))

    if not directed:
        w = tf.maximum(w, tf.transpose(w, [0,2,1]))

    mask = tf.where(tf.greater_equal(w, 1e-5), tf.ones_like(w), tf.zeros_like(w))
    w = tf.multiply(w, mask)

    if e is None:
        wdx = dx
    else:
        wdx = tf.cast(tf.less_equal(e, eps), tf.float32  if not use_fp16 else tf.float16) * (1e10  if not use_fp16 else 1e5) + dx

    return w, wdx

def edgeattention(gs, sgs, labelednum, unlabelednum, thres=2.0):
    batchsize = tf.shape(gs)[0]
    absdy = tf.math.abs(sgs - gs)
    m = tf.less_equal(absdy, thres)
    mask = tf.equal(tf.expand_dims(m, 1), tf.expand_dims(m,2))
    mask = tf.where(mask, tf.ones_like(mask, tf.float32), tf.zeros_like(mask, tf.float32))
    return tf.slice(tf.cast(tf.logical_not(m), tf.float32), [0, labelednum], [batchsize, unlabelednum]), mask

def glr_fidelity(a, y, mu, labelednum, unlabelednum, nodenum,  ally=False, normlap=False, kappafactor=1.0):
    batchsize = tf.shape(a)[0]

    A = a - tf.linalg.diag(tf.linalg.diag_part(a))
    D = tf.reduce_sum(A, axis=2)
    if normlap:
        L = norm_lap(A)
    else:
        L = tf.linalg.diag(D) - A

    if ally:
        gs = tf.reshape(y, (batchsize, nodenum, 1))
    else:
        gs = tf.reshape(tf.concat((y, tf.zeros((batchsize, unlabelednum))), axis=1), (batchsize, nodenum, 1))

    if mu is None:
        kappa = 60.0 * kappafactor
        mu_scale = 0.6667
        mu_max = (kappa-1.0)/(2.0* tf.reduce_max(D, axis=1, keepdims=True))
        mu_max = tf.expand_dims(mu_max, -1)
        mu = mu_scale * mu_max

    I = tf.eye(nodenum, batch_shape=[batchsize])

    sgs = tf.linalg.solve(I+mu*L, gs)

    sgs = tf.reshape(sgs, (batchsize, nodenum))

    if not ally:
        labeled_gs = tf.slice(y, [0,0], [batchsize, labelednum])
        labeled_sgs = tf.slice(sgs, [0,0], [batchsize, labelednum])

        all_gs = tf.concat((labeled_gs, 1000.0*tf.ones((batchsize, unlabelednum), tf.float32)), axis=1)
        all_sgs = tf.concat((labeled_sgs, -1000.0*tf.ones((batchsize, unlabelednum), tf.float32)), axis=1)
    else:
        all_gs = y
        all_sgs = sgs

    return sgs, all_gs, all_sgs

def postprocess_grads(grads, **kwargs):
    clip_norm = kwargs.get("clip_norm", None)
    grads = [None if grad is None else tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in grads]
    grads = [None if grad is None else tf.where(tf.math.is_inf(grad), tf.zeros_like(grad), grad) for grad in grads]
    if clip_norm is not None:
        grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm) for grad in grads]
    return grads

#x: batchsize*nodenum*nodenum (dists)
#out: idx = batchsize*nodenum*k
#     mask = batchsize*nodenum*k
def knn1d(x, nodenum, batchsize, k=6, eps=1e-7):
    #remove selfloop edge
    noselfx = tf.linalg.set_diag(x, -1e10*tf.ones((batchsize, nodenum)))
    val, idx = tf.nn.top_k(noselfx, k=k)
    mask = tf.cast(tf.greater(val, eps), tf.float32)
    return idx, mask

#x: batchsize*nodenum*featdim (features)
#mau: batchsize*nodenum
#out: batchsize*nodenum*(k+1)*featdim
def knnfeature(x, idx, mask, batchsize, nodenum, featdim, k=6, mau=None):
    bidx = tf.range(batchsize) * nodenum
    bidx = tf.reshape(bidx, [batchsize, 1, 1])

    gx = tf.reshape(x, [-1, featdim])
    nx = tf.gather(gx, idx+bidx)
    cx = tf.expand_dims(x, axis=-2)
    tcx = tf.tile(cx, [1, 1, k, 1])

    nf = nx-tcx   #nf=batchsize*nodenum*k*featdim
    m = tf.expand_dims(mask, axis=-1)  #m=batchsize*nodenum*k*1
    mnf = tf.multiply(m, nf)

    mm = tf.clip_by_value(mau, 0, k)
    mm = tf.sequence_mask(mm, k, tf.float32)
    mnf = tf.multiply(mnf, tf.expand_dims(mm, 3))

    return tf.transpose(tf.concat([cx, mnf], axis=-2), [0, 1, 3, 2])

def sparse_knn(dx, k, nodenum, eps=1e-7):
    batchsize = tf.shape(dx)[0]
    nodenum = tf.shape(dx)[1]

    bindices = tf.tile(tf.expand_dims(tf.range(batchsize)* nodenum * nodenum, 1), (1, nodenum*nodenum))
    gindices = tf.reshape(tf.tile(tf.expand_dims(tf.range(nodenum) * nodenum, 1),(batchsize, nodenum)), [batchsize,-1])
    gindices = gindices + bindices

    dx = tf.linalg.set_diag(dx, 1e10*tf.ones([batchsize, nodenum]))
    valid_mask =tf.cast(tf.less(dx, 1e8), tf.float32)

    sortedinds = tf.argsort(dx, 2, direction='ASCENDING', stable=True)
    seqmask = tf.sequence_mask(k, nodenum, tf.int32)
    sortedinds = sortedinds *seqmask + (seqmask - 1)*1000000
    sortedginds =  tf.reshape(tf.reshape(sortedinds, (batchsize, nodenum*nodenum))+gindices, (-1,))

    bool_mask = tf.greater_equal(sortedginds, 0)
    sortedginds = tf.reshape(tf.boolean_mask(sortedginds, bool_mask), (-1,1))

    sparse_mask  = tf.reshape(tf.scatter_nd(sortedginds, tf.ones(tf.shape(sortedginds), tf.float32), tf.shape(tf.reshape(dx,[-1,1]))), tf.shape(dx)) * valid_mask

    sparse_mask = tf.math.maximum(sparse_mask, tf.transpose(sparse_mask, [0,2,1]))

    return sparse_mask

def res_net_block(input_data, filters, conv_size):
  x = tf.keras.layers.Conv1D(filters, conv_size, activation=None, padding='same')(input_data)
  x = tf.keras.layers.Add()([x, input_data])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def dynglr(**kwargs):
    #graph weighting1
    inputs = tf.keras.Input(shape=(kwargs.get("fc_inputdim", 2048),))
    x =  tf.keras.layers.Reshape((32, int(kwargs.get("fc_inputdim", 2048)/32)))(inputs)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu', use_bias=False)(x)
    shallows = tf.keras.layers.MaxPooling1D(2)(x)
    x = res_net_block(shallows, 256, 3)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(32)(x)
    model_gw1 = tf.keras.Model(inputs, [outputs, shallows])

    #graph update
    tau = kwargs.get('dynglr_tau', 3)
    inputs1 = tf.keras.Input(shape=(15, 256))  #node features
    inputs2 = tf.keras.Input(shape=((tau+1)*2,))   #tau neigbors' labels
    x1 = tf.keras.layers.Conv1D(112, 3, activation='relu', use_bias=False)(inputs1)
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
    x2 = tf.keras.layers.Dense(16, activation='relu', use_bias=False)(inputs2)
    shallows1 = tf.keras.layers.Concatenate()([x1, x2])
    x12 = tf.keras.layers.Reshape((16, 8))(shallows1)
    x12 = tf.keras.layers.Conv1D(128, 3, activation='relu', use_bias=False)(x12)
    x12 = tf.keras.layers.GlobalAveragePooling1D()(x12)
    outputs1 = tf.keras.layers.Add()([x12, shallows1])
    model_gu = tf.keras.Model([inputs1, inputs2], [outputs1, shallows1])

    #graph weighting2
    inputs3 = tf.keras.Input(shape=(15, 256))
    inputs4 = tf.keras.Input(shape=(128,))

    x3 = tf.keras.layers.Conv1D(32, 3, activation='relu', use_bias=False)(inputs3)
    x3 = tf.keras.layers.GlobalAveragePooling1D()(x3)
    x4 = tf.keras.layers.Reshape((16, 8))(inputs4)
    x4 = tf.keras.layers.Conv1D(64, 3, activation='relu', use_bias=False)(x4)
    x4 = tf.keras.layers.GlobalAveragePooling1D()(x4)

    outputs2 = tf.keras.layers.Concatenate()((x3, x4))
    model_gw2 = tf.keras.Model([inputs3, inputs4], outputs2)

    return model_gw1, model_gu, model_gw2


def dynglr_knnfeature(y, wd, tau, mau=None):
    batchsize = tf.shape(y)[0]
    nodenum = tf.shape(y)[1]

    if tau is not None and tau!=0:
        idx, mask =  knn1d(wd, nodenum, batchsize, tau)
        fy = knnfeature(y, idx, mask, batchsize, nodenum, 2, tau, mau)
    else:
        fy = y

    return fy

def tf_arch(classes, **kwargs):
    return dynglr(**kwargs)


def is_sequence(obj):
    return hasattr(type(obj), '__iter__')

def lr_constant(lr, step):
    return lr

def lr_piecewiseconstant(lrs, stairs, step):
    stair_s = -1
    for i in range(len(stairs)):
        if step<stairs[i]:
            stair_s = i-1
            break
    if stair_s == -1 or stair_s>=len(stairs):
        return lrs[-1]
    ret =lrs[stair_s] +  (step - stairs[stair_s])*(lrs[stair_s+1] - lrs[stair_s]) / (stairs[stair_s+1] - stairs[stair_s])
    return ret

def lr_piecewiselinear(knots, vals, step):
    return np.interp(step, knots, vals)

def get_arch_lr(**kwargs):
    lr_mode= kwargs.get("lr_mode", 'constant')
    if lr_mode=='constant':
        return partial(lr_constant, kwargs.get("lr_init", 0.001))
    elif lr_mode=='piecewise_constant':
        return partial(lr_piecewiseconstant, kwargs.get("lrs", [0.001, 0.0005, 0.0001, 0.0005, 0.0001, 0.00005]), kwargs.get("lr_stairs", [40, 80, 140, 220, 320, 440]))
    elif lr_mode=='piecewise_linear':
        return partial(lr_piecewiselinear, kwargs.get("lr_knots", [0, 2, 8]), kwargs.get("lr_vals", [0, 1.0, 0.1]))

class CustomLoss2(object):
  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
    losses_utils.ReductionV2.validate(reduction)
    self.reduction = reduction
    self.name = name

  def __call__(self, y_true, y_pred, w, sample_weight=None):
    scope_name = 'lambda' if self.name == '<lambda>' else self.name
    graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
        y_true, y_pred, w, sample_weight)
    with K.name_scope(scope_name or self.__class__.__name__), graph_ctx:
      losses = self.call(y_true, y_pred, w)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self._get_reduction())

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    return {'reduction': self.reduction, 'name': self.name}

  def call(self, y_true, y_pred, w):
    NotImplementedError('Must be implemented in subclasses.')

  def _get_reduction(self):
    if distribution_strategy_context.has_strategy() and (
        self.reduction == losses_utils.ReductionV2.AUTO or
        self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE):
      raise ValueError(
          'Please use `tf.keras.losses.Reduction.SUM` or '
          '`tf.keras.losses.Reduction.NONE` for loss reduction when losses are '
          'used with `tf.distribute.Strategy` outside of the built-in training '
          'loops. You can implement '
          '`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch '
          'size like:\n```\nwith strategy.scope():\n'
          '    loss_obj = tf.keras.losses.CategoricalCrossentropy('
          'reduction=tf.keras.losses.reduction.NONE)\n....\n'
          '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * '
          '(1. / global_batch_size)\n```\nPlease see '
          'https://www.tensorflow.org/alpha/tutorials/distribute/training_loops'
          ' for more details.')

    if self.reduction == losses_utils.ReductionV2.AUTO:
      return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
    return self.reduction

class CustomLossFunctionWrapper2(CustomLoss2):
  def __init__(self, fn, reduction=losses_utils.ReductionV2.AUTO, name=None, **kwargs):
    super(CustomLossFunctionWrapper2, self).__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred, w):
    if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true) and tensor_util.is_tensor(w):
      y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(
          y_pred, y_true)
    return self.fn(y_true, y_pred, w, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    base_config = super(CustomLossFunctionWrapper2, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def tripletloss(y, dx, **kwargs):
    alpha = kwargs.get('margin', 10.0)
    T = tf.cast(tf.equal(tf.expand_dims(y,1), tf.expand_dims(y, 2)), tf.float32)
    F = tf.subtract(1.0, T)
    P = tf.multiply(T, dx)
    N = tf.multiply(F, tf.nn.relu(tf.subtract(alpha, dx)))
    return  tf.reduce_mean(P+N)

def tripletglrloss(y, dx, w, **kwargs):
    alpha = kwargs.get('margin', 10.0)
    T = tf.cast(tf.equal(tf.expand_dims(y,1), tf.expand_dims(y, 2)), tf.float32)
    F = tf.subtract(1.0, T)
    P = tf.multiply(T, dx)
    N = tf.multiply(F, tf.nn.relu(tf.subtract(alpha, dx)))
    return  tf.reduce_mean(tf.multiply(P+N, w))

class TripletGLRLossError(CustomLossFunctionWrapper2):
  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='tripletglrloss_error', **kwargs):
    super(TripletGLRLossError, self).__init__(tripletglrloss, name=name, reduction=reduction, **kwargs)

class CustomMeanMetricWrapper2(tf.keras.metrics.Mean):
  def __init__(self, fn, name=None, dtype=None, **kwargs):
    super(CustomMeanMetricWrapper2, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, w, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    w = math_ops.cast(w, self._dtype)

    [y_true, y_pred], sample_weight = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight)

    y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(
        y_pred, y_true)

    matches = self._fn(y_true, y_pred, w, **self._fn_kwargs)
    return super(CustomMeanMetricWrapper2, self).update_state(
        matches, sample_weight=sample_weight)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(CustomMeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class TripletGLRLoss(CustomMeanMetricWrapper2):
  def __init__(self, name='tripletglrloss', dtype=None, **kwargs):
    super(TripletGLRLoss, self).__init__(tripletglrloss, name, dtype=dtype, **kwargs)