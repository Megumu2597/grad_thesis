# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import time
import tensorflow as tf
import scipy.signal
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, concatenate,BatchNormalization, Activation, Convolution2D, MaxPooling2D, Flatten, Input,LeakyReLU, Lambda
#from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K
from collections import deque

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_render_func
#from ..trained_visionmodel.visionmodel import VisionModelXYZ, VisionModel

from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.core import logger
from rlkit.envs.wrappers import NormalizedBoxEnv
import uuid
import doorenv
import doorenv2
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32


def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def gauss_prob_val(mu, logstd, x):
    std = np.exp(logstd)
    var = np.square(std)
    gp = np.exp(-np.square(x - mu)/(2*var)) / ((2*np.pi)**.5 * std)
    return np.prod(gp, axis=1)

def gauss_prob(mu, logstd, x):
    std = tf.exp(logstd)
    var = tf.square(std)
    gp = tf.exp(-tf.square(x - mu)/(2*var)) / ((2*np.pi)**.5 * std)
    return tf.reduce_prod(gp, [1])

def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2*logstd)
    gp = -tf.square(x - mu)/(2 * var) - .5*tf.log(tf.constant(2*np.pi)) - logstd
    return tf.reduce_sum(gp, [1])

def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd
    return gauss_KL(mu1, logstd1, mu2, logstd2)

def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2*logstd1)
    var2 = tf.exp(2*logstd2)
    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2))/(2*var2) - 0.5)
    return kl

def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5*np.log(2*np.pi*np.e), tf.float32))
    return h

def gauss_sample(mu, logstd):
    return mu + tf.exp(logstd)*tf.random_normal(tf.shape(logstd))

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
            "shape function assumes that shape is fully known"
    return out

def numel(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat( [tf.reshape(grad, [numel(v)])#自作ほんとは0は第１引数にあった
                         for (v, grad) in zip(var_list, grads)],0)

def rollout_contin(env, generator, action_dim, aux_dim, encode_dim,
                   max_step_limit, min_step_limit,  paths_per_collect,
                   pre_actions, posterior,statedemo):
    paths = []
    timesteps_sofar = 0
    encode_axis = 0
    #env_obj = env.venv.venv.envs[0].env.env
    #print(pre_actions.shape)
    for p in range(paths_per_collect):
        print ("Rollout index:", p)
        auxs, encodes,  raws, actions, logstds = [], [], [], [], []

        encode = np.zeros((1, encode_dim), dtype=np.float32)
        encode[0, encode_axis] = 1
        encode_axis = (encode_axis + 1) % encode_dim
        print ("Encode:", encode)
        #reward_p = 0
        for i in range(max_step_limit):
            aux=statedemo[i+512*p].reshape(1,-1)#初期化に相当
            auxs.append(aux)
            encodes.append(encode)
            logstd = np.array([[-20, -20, -20,-20,-20,-20]], dtype=np.float32)
            logstds.append(logstd)
            action = generator.predict([ aux, encode]) #[1,6]
            actions.append(action)
            raw = [1]#reward ppoi??
            raws.append(raw)
            #reward_p += np.sum(np.log(posterior.predict([ aux, action]))* encode)

            if  i + 1 == max_step_limit:#resDONEs
                path = dict2(auxs = np.concatenate(auxs),
                             encodes = np.concatenate(encodes),
                             actions = np.concatenate(actions),
                             logstds = np.concatenate(logstds),
                             raws = np.array(raws))
                paths.append(path)
                step = i + 1 - min_step_limit - pre_step
                print ("Step:", step,  )#"Reward_p:", reward_p )
                break

            res = env.step(action[0])


    #env.end()
    env.close()
    return paths


class LinearBaseline(object):
    coeffs = None

    def _features(self, path):
        o = path["states"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al, al**2, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + lamb * np.identity(n_col),
            featmat.T.dot(returns))[0]

    def predict(self, path):
        return np.zeros(len(path["rewards"])) if self.coeffs is None else \
                self._features(path).dot(self.coeffs)


def pathlength(path):
    return len(path["actions"])

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


class TimeDependentBaseline(object):
    def __init__(self):
        self.baseline = None

    def fit(self, paths):
        rets = [path["returns"] for path in paths]
        maxlen = max(len(ret) for ret in rets)
        retsum = np.zeros(maxlen)
        retcount = np.zeros(maxlen)
        for ret in rets:
            retsum[:len(ret)] += ret
            retcount[:len(ret)] += 1
        retmean = retsum / retcount
        self.baseline = retmean
        pred = np.concatenate([self.predict(path) for path in paths])
        return {"EV" : explained_variance(pred, np.concatenate(rets))}

    def predict(self, path):
        if self.baseline is None:
            return np.zeros(pathlength(path))
        else:
            lenpath = pathlength(path)
            lenbase = len(self.baseline)
            if lenpath > lenbase:
                return np.concatenate([self.baseline, self.baseline[-1] +
                                       np.zeros(lenpath-lenbase)])
            else:
                return self.baseline[:lenpath]


class NNBaseline(object):
    def __init__(self, sess,  aux_dim, encode_dim, lr_baseline,
                 b_iter, batch_size):
        print ("Now we build baseline")
        self.model = self.create_net( aux_dim, encode_dim, lr_baseline)
        self.sess = sess
        self.b_iter = b_iter
        self.batch_size = batch_size
        self.first_time = True
        self.mixfrac = 0.1

    def create_net(self,  aux_dim, encode_dim, lr_baseline):

        auxs = Input(shape=[aux_dim + 1])
        encodes = Input(shape=[encode_dim])
        #h = merge([x, auxs, encodes], mode='concat')
        h = concatenate([auxs, encodes],axis=-1)
        h = Dense(64)(h)#256
        h = LeakyReLU()(h)
        h = Dense(64)(h)#128
        h = LeakyReLU()(h)
        p = Dense(1)(h)

        model = Model(inputs=[ auxs, encodes], outputs=p)
        adam = Adam(lr=lr_baseline)
        model.compile(loss='mse', optimizer=adam)
        return model

    def _get_aux(self, path):
        # Add time stamp
        aux = path["auxs"].astype('float32')
        aux = aux.reshape(aux.shape[0], -1)
        l = len(path["auxs"])
        al = np.arange(l).reshape(-1, 1) / 512.0 #100自作
        return np.concatenate([aux, al], axis=1)

    def fit(self, paths):
        auxs = np.concatenate([self._get_aux(path) for path in paths])
        encodes = np.concatenate([path["encodes"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        if self.first_time:
            self.first_time = False
            b_iter = 100 #100#自作
        else:
            returns_old = np.concatenate([self.predict(path) for path in paths])
            returns = returns * self.mixfrac + returns_old * (1 - self.mixfrac)
            b_iter = self.b_iter

        num_data = auxs.shape[0]
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        train_val_ratio = 0.7
        num_train = int(num_data * train_val_ratio)
        auxs_train = auxs[idx][:num_train]
        encodes_train = encodes[idx][:num_train]
        returns_train = returns[idx][:num_train]

        auxs_val = auxs[idx][num_train:]
        encodes_val = encodes[idx][num_train:]
        returns_val = returns[idx][num_train:]

        start = 0
        batch_size = self.batch_size
        for i in range(b_iter):
            loss = self.model.train_on_batch(
                [auxs_train[start:start + batch_size],
                 encodes_train[start:start + batch_size]],
                returns_train[start:start + batch_size]
            )
            start += batch_size
            if start >= num_train:
                start = (start + batch_size) % num_train
            val_loss = np.average(np.square(self.model.predict(
                [ auxs_val, encodes_val]).flatten() - returns_val))
            print ("Baseline step:", i, "loss:", loss, "val:", val_loss)

    def predict(self, path):
        if self.first_time:
            return np.zeros(pathlength(path))
        else:
            ret = self.model.predict(
                [ self._get_aux(path), path["encodes"]])
        return np.reshape(ret, (ret.shape[0], ))



class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat( [tf.reshape(v, [numel(v)]) for v in var_list],0)#自作0を1から2引数に

    def __call__(self):
        return self.op.eval(session=self.session)


class SetFromFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(v, tf.reshape(theta[start:start + size], shape))
            )
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_paths = 0
        self.buffer = deque()

    def get_sample(self, sample_size):
        if self.num_paths < sample_size:
            return random.sample(self.buffer, self.num_paths)
        else:
            return random.sample(self.buffer, sample_size)

    def size(self):
        return self.buffer_size

    def add(self, path):
        if self.num_paths < self.buffer_size:
            self.buffer.append(path)
            self.num_paths += 1
        else:
            self.buffer.popleft()
            self.buffer.append(path)

    def count(self):
        return self.num_paths

    def erase(self):
        self.buffer = deque()
        self.num_paths = 0
