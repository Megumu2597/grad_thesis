# -*- coding: utf-8 -*-
from proputils import *
import numpy as np
import time
import math
import argparse
from tensorflow.keras.initializers import normal, identity, uniform,VarianceScaling,RandomNormal
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.engine.topology import Container
from tensorflow.keras.layers import Dense,LeakyReLU,BatchNormalization, Activation, Convolution2D, MaxPooling2D, Flatten, Input,  Lambda
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import concatenate,add
#K.set_image_dim_ordering('tf')
import json
import matplotlib
matplotlib.use("Agg") ##GUIない環境では必要
import matplotlib.pyplot as plt

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

from tensorflow.keras.applications.resnet50 import ResNet50
"""from keras.initializations import normal, identity, uniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import keras.backend as K
"""


parser = argparse.ArgumentParser(description="TRPO")
parser.add_argument("--paths_per_collect", type=int, default=10)
parser.add_argument("--max_step_limit", type=int, default=513)#300kokoga timestep ni soutou
parser.add_argument("--min_step_limit", type=int, default=0)#origin:100,(2)始まるまで動かないTODO適切な値
#parser.add_argument("--pre_step", type=int, default=1)#100,(100)prestep
parser.add_argument("--n_iter", type=int, default=5001) #1000 38以上 epoch
parser.add_argument("--gamma", type=float, default=.95)
parser.add_argument("--lam", type=float, default=.97)
parser.add_argument("--max_kl", type=float, default=0.01)
parser.add_argument("--cg_damping", type=float, default=0.1)
parser.add_argument("--lr_discriminator", type=float, default=5e-5)#5
parser.add_argument("--d_iter", type=int, default=100)#100
parser.add_argument("--clamp_lower", type=float, default=-0.01)
parser.add_argument("--clamp_upper", type=float, default=0.01)
parser.add_argument("--lr_baseline", type=float, default=1e-4)#4
parser.add_argument("--b_iter", type=int, default=25)#25
parser.add_argument("--lr_posterior", type=float, default=1e-4)#4
parser.add_argument("--p_iter", type=int, default=40)#50
parser.add_argument("--buffer_size", type=int, default=300)#75
parser.add_argument("--sample_size", type=int, default=200)#50
parser.add_argument("--batch_size", type=int, default=5120)#512

args = parser.parse_args()


class TRPOAgent(object):
    config = dict2(paths_per_collect = args.paths_per_collect,
                   max_step_limit = args.max_step_limit,#1epiのtime step num
                   min_step_limit = args.min_step_limit,
                   n_iter = args.n_iter,
                   gamma = args.gamma,
                   lam = args.lam,
                   max_kl = args.max_kl,
                   cg_damping = args.cg_damping,
                   lr_discriminator = args.lr_discriminator,
                   d_iter = args.d_iter,
                   clamp_lower = args.clamp_lower,
                   clamp_upper = args.clamp_upper,
                   lr_baseline = args.lr_baseline,
                   b_iter = args.b_iter,
                   lr_posterior = args.lr_posterior,
                   p_iter = args.p_iter,
                   buffer_size = args.buffer_size,
                   sample_size = args.sample_size,
                   batch_size = args.batch_size)

    def __init__(self, env, sess,  aux_dim, encode_dim, action_dim, pre_actions):
        self.env = env
        self.sess = sess
        self.buffer = ReplayBuffer(self.config.buffer_size)

        self.aux_dim = aux_dim
        self.encode_dim = encode_dim
        self.action_dim = action_dim
        self.pre_actions = pre_actions
        self.auxs = auxs = tf.placeholder(dtype, shape=[None, aux_dim])
        self.encodes = encodes = tf.placeholder(dtype, shape=[None, encode_dim])
        self.actions = actions = tf.placeholder(dtype, shape=[None, action_dim])

        self.advants = advants = tf.placeholder(dtype, shape=[None])
        self.oldaction_dist_mu = oldaction_dist_mu = \
                tf.placeholder(dtype, shape=[None, action_dim])
        self.oldaction_dist_logstd = oldaction_dist_logstd = \
                tf.placeholder(dtype, shape=[None, action_dim])

        # Create neural network.
        print ("Now we build trpo generator")
        self.generator = self.create_generator( auxs, encodes)
        print ("Now we build posterior")
        self.posterior = \
            self.create_posterior( aux_dim, action_dim, encode_dim)
        self.posterior_target = \
            self.create_posterior( aux_dim, action_dim, encode_dim)

        self.demo_idx = 0

        action_dist_mu = self.generator.outputs[0]
        # self.action_dist_logstd_param = action_dist_logstd_param = \
        #         tf.placeholder(dtype, shape=[1, action_dim])
        # action_dist_logstd = tf.tile(action_dist_logstd_param,
        #                              tf.pack((tf.shape(action_dist_mu)[0], 1)))
        action_dist_logstd = tf.placeholder(dtype, shape=[None, action_dim])

        eps = 1e-8
        self.action_dist_mu = action_dist_mu
        self.action_dist_logstd = action_dist_logstd
        N = tf.shape(actions)[0]
        # compute probabilities of current actions and old actions
        log_p_n = gauss_log_prob(action_dist_mu, action_dist_logstd, actions)
        log_oldp_n = gauss_log_prob(oldaction_dist_mu, oldaction_dist_logstd, actions)

        ratio_n = tf.exp(log_p_n - log_oldp_n)
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advants) # Surrogate loss
        var_list = self.generator.trainable_weights

        kl = gauss_KL(oldaction_dist_mu, oldaction_dist_logstd,
                      action_dist_mu, action_dist_logstd) / Nf
        ent = gauss_ent(action_dist_mu, action_dist_logstd) / Nf

        self.losses = [surr, kl, ent]
        #print(self.losses)
        #self.sess.run(self.losses[0])
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        kl_firstfixed = gauss_selfKL_firstfixed(action_dist_mu,
                                                action_dist_logstd) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.sess, var_list)
        self.sff = SetFromFlat(self.sess, var_list)
        self.baseline = NNBaseline(sess, aux_dim, encode_dim,
                                   self.config.lr_baseline, self.config.b_iter,
                                   self.config.batch_size)
        self.sess.run(tf.global_variables_initializer())

    def create_generator(self, auxs, encodes):

        auxs = Input(tensor=auxs)
        #h = concatenate([x, auxs], axis=-1)
        encodes = Input(tensor=encodes)
        h = concatenate([auxs, encodes],axis=-1)
        h = Dense(64)(h)
        h = LeakyReLU()(h)
        h = Dense(64)(h)
        h = LeakyReLU()(h)

        actions = Dense(6, activation='tanh')(h)
        #actions = merge([act1,act2,act3,act4,act5,act6], mode='concat')
        #actions = merge([act1, act2, act3], mode='concat')
        #model = Model(inputs=[fdim,adim,edim], outputs=actions
        model = Model(inputs=[ auxs, encodes], outputs=actions)
        #####自作
        model.summary()
        return model

    def create_posterior(self,  aux_dim, action_dim, encode_dim):

        auxs = Input(shape=[aux_dim])
        actions = Input(shape=[action_dim])
        #h = merge([x, auxs, actions], mode='concat')
        h = concatenate([auxs, actions], axis=-1)
        h = Dense(64)(h)#256
        h = LeakyReLU()(h)
        h = Dense(64)(h)#128
        h = LeakyReLU()(h)
        c = Dense(encode_dim, activation='softmax')(h)

        model = Model(inputs=[auxs, actions], outputs=c)
        adam = Adam(lr=self.config.lr_posterior)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=['accuracy'])
        return model

    def create_detector(self,  aux_dim, action_dim, encode_dim):
        states  = Input(shape=[aux_dim])
        actions = Input(shape=[action_dim])
        p = concatenate([states,actions],axis=-1)
        p = Dense(64)(p)#256
        p = LeakyReLU()(p)
        p = Dense(64)(p)#128
        p = LeakyReLU()(p)
        c = Dense(aux_dim, activation='softmax')(p)
        #encoder ここまで
        input_d = concatenate([states_p,c],axis=-1)
        d = Dense(64)(input_d)#256
        p = LeakyReLU()(d)
        d = Dense(64)(d)#128
        d = LeakyReLU()(d)
        a = Dense(action_dim, activation='tanh')(d)
        #decoderここまで
        #states = Input(shape=[aux_dim])
        #actions = Input(shape=[action_dim])
        #encodes = Input(shape=[encode_dim])
        h = concatenate([states,actions,c],axis=-1)
        h = Dense(64)(h)#256
        h = LeakyReLU()(h)
        h = Dense(64)(h)#128
        h = LeakyReLU()(h)
        s = Dense(aux_dim, activation='tanh')(h)
        detector = Model(inputs=[states,actions,encodes],outputs=c)
        adam = Adam(lr=self.config.lr_posterior)
        detector.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=['accuracy'])

        return detector


    def predict(self, auxs, encodes, logstds, *args):
        action_dist_mu = \
                self.sess.run(
                    self.action_dist_mu,
                    {self.auxs: auxs, self.encodes: encodes}
                )

        act = action_dist_mu + np.exp(logstds) * \
                np.random.randn(*logstds.shape)
        #TODO
        act[:, 0] = np.clip(act[:, 0], -1, 1)
        act[:, 1] = np.clip(act[:, 1], -1, 1)
        act[:, 2] = np.clip(act[:, 2], -1, 1)
        act[:, 3] = np.clip(act[:, 3], -1, 1)
        act[:, 4] = np.clip(act[:, 4], -1, 1)
        act[:, 5] = np.clip(act[:, 5], -1, 1)

        return act

    def learn(self, demo):
        config = self.config
        start_time = time.time()
        numeptotal = 0
        ave_rewards=[]

        # Set up for training discrimiator
        print ("Loading data ...")
        auxs_d, actions_d =  demo["obs"], demo["actions"]
        #1エピソードごとにシャッフル
        actions_d = actions_d.reshape(-1,512,self.action_dim)
        auxs_d    = auxs_d.reshape(-1,512,self.aux_dim)
        num_epi= actions_d.shapce[0]
        idx_d = np.arange(num_epi)
        np.random.shuffle(idx_d)

        auxs_d = auxs_d[idx_d]
        actions_d = actions_d[idx_d]
        print ("Resizing data for demo ...")
        actions_d = actions_d.reshape(-1,self.action_dim)
        auxs_d    = auxs_d.reshape(-1,self.aux_dim)
        numdetotal = actions_d.shapce[0]

        for i in range(0, config.n_iter):

            # Generating paths.
            # if i == 1:
            print("i= ",i)
            paths_per_collect = num_epi
            rollouts = rollout_contin(
                self.env,
                self,
                self.action_dim,
                self.aux_dim,
                self.encode_dim,
                config.max_step_limit,
                config.min_step_limit,
                paths_per_collect,
                self.pre_actions,
                self.posterior_target,auxs_d)

            for path in rollouts:
                self.buffer.add(path)
            print ("Buffer count:", self.buffer.count())
            paths = self.buffer.get_sample(config.sample_size)

            print ("Calculating actions ...")
            for path in paths:
                path["mus"] = self.sess.run(
                    self.action_dist_mu,
                    { self.auxs: path["auxs"],
                     self.encodes: path["encodes"]}
                )

            mus_n = np.concatenate([path["mus"] for path in paths])
            logstds_n = np.concatenate([path["logstds"] for path in paths])
            auxs_n = np.concatenate([path["auxs"] for path in paths])
            encodes_n = np.concatenate([path["encodes"] for path in paths])
            actions_n = np.concatenate([path["actions"] for path in paths])
            print ("Epoch:", i, "Total sampled data points:", actions_n.shape[0])
            # Train discriminator
            numnototal = actions_n.shape[0]
            batch_size = config.batch_size
            start_d = self.demo_idx
            start_n = 0
            if i <= 5:
                d_iter = 120 - i* 20#i*20自作
            else:
                d_iter = 10
            #i = i+300#######調整用

            for k in range(d_iter):
                loss = self.discriminator.train_on_batch(
                    [auxs_n[start_n:start_n + batch_size],
                     actions_n[start_n:start_n + batch_size],
                     auxs_d[start_d:start_d + batch_size],
                     actions_d[start_d:start_d + batch_size]],
                    np.ones(batch_size)
                )

                # print self.discriminator.summary()
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, config.clamp_lower, config.clamp_upper)
                               for w in weights]
                    l.set_weights(weights)

                start_d = self.demo_idx = self.demo_idx + batch_size
                start_n = start_n + batch_size

                if start_d + batch_size >= numdetotal:
                    start_d = self.demo_idx = (start_d + batch_size) % numdetotal
                if start_n + batch_size >= numnototal:
                    start_n = (start_n + batch_size) % numnototal

                print ("Discriminator step:", k, "loss:", loss)
            loss_d=loss #自作
            idx = np.arange(numnototal)
            #print("numnototal demonosankouni",numnototal)
            np.random.shuffle(idx)
            train_val_ratio = 0.7
            # Training data for posterior
            numno_train = int(numnototal * train_val_ratio)
            auxs_train = auxs_n[idx][:numno_train]
            actions_train = actions_n[idx][:numno_train]
            encodes_train = encodes_n[idx][:numno_train]
            # Validation data for posterior
            auxs_val = auxs_n[idx][numno_train:]
            actions_val = actions_n[idx][numno_train:]
            encodes_val = encodes_n[idx][numno_train:]

            start_n = 0
            for j in range(config.p_iter):
                loss = self.posterior.train_on_batch(
                    [auxs_train[start_n:start_n + batch_size],
                     actions_train[start_n:start_n + batch_size]],
                    encodes_train[start_n:start_n + batch_size]
                )
                start_n += batch_size
                if start_n + batch_size >= numno_train:
                    start_n = (start_n + batch_size) % numno_train

                posterior_weights = self.posterior.get_weights()
                posterior_target_weights = self.posterior_target.get_weights()
                for k in range(len(posterior_weights)):
                    posterior_target_weights[k] = 0.5 * posterior_weights[k] +\
                            0.5 * posterior_target_weights[k]
                self.posterior_target.set_weights(posterior_target_weights)

                output_p = self.posterior_target.predict( [ auxs_val, actions_val])
                output_p = np.where(output_p<1e-4,1e-4,output_p)
                val_loss = -np.average( np.sum(np.log(output_p) * encodes_val, axis=1))
                print ("Posterior step:", j, "loss:", loss, val_loss)
            loss_p,val_loss_p = loss,val_loss#自作
            # Computing returns and estimating advantage function.
            path_idx = 0
            maxrewards = []
            for path in paths:
                #TODO
                #file_path = "../DoorGym/bcmodel/part/iter_%d_path_%d.txt" % (i, path_idx)
                #f = open(file_path, "w")
                path["baselines"] = self.baseline.predict(path)
                output_d = self.discriminate.predict(
                    [ path["auxs"], path["actions"]])
                output_p = self.posterior_target.predict(
                    [ path["auxs"], path["actions"]])
                output_p = np.where(output_p<1e-4,1e-4,output_p) #自作
                path["rewards"] = np.ones(path["raws"].shape[0]) * 2 + \
                        output_d.flatten() * 0.1 + \
                        np.sum(np.log(output_p) * path["encodes"], axis=1)

                path_baselines = np.append(path["baselines"], 0 if
                                           path["baselines"].shape[0] == 100 else
                                           path["baselines"][-1])
                deltas = path["rewards"] + config.gamma * path_baselines[1:] -\
                        path_baselines[:-1]
                # path["returns"] = discount(path["rewards"], config.gamma)
                # path["advants"] = path["returns"] - path["baselines"]
                path["advants"] = discount(deltas, config.gamma * config.lam)
                path["returns"] = discount(path["rewards"], config.gamma)
                """
                f.write("Baseline:\n" + np.array_str(path_baselines) + "\n")
                f.write("Returns:\n" + np.array_str(path["returns"]) + "\n")
                f.write("Advants:\n" + np.array_str(path["advants"]) + "\n")
                f.write("Mus:\n" + np.array_str(path["mus"]) + "\n")
                f.write("Actions:\n" + np.array_str(path["actions"]) + "\n")
                f.write("Logstds:\n" + np.array_str(path["logstds"]) + "\n")
                f.close()"""
                path_idx += 1
                maxrewards.append(np.amax(path["returns"]).tolist())

            rewards_ave = sum(maxrewards) / len(maxrewards)
            # Standardize the advantage function to have mean=0 and std=1
            advants_n = np.concatenate([path["advants"] for path in paths])
            # advants_n -= advants_n.mean()
            advants_n /= (advants_n.std() + 1e-8)

            #print("shape ,aux,encode,act,acvants,actdist,olda,oldaclo",auxs_n.shape,encodes_n.shape,actions_n.shape,advants_n.shape,logstds_n.shape,mus_n.shape)
            #####自作
            # Computing baseline function for next iter.
            self.baseline.fit(paths)

            feed = {self.auxs: auxs_n,
                    self.encodes: encodes_n,
                    self.actions: actions_n,
                    self.advants: advants_n,
                    self.action_dist_logstd: logstds_n,
                    self.oldaction_dist_mu: mus_n,
                    self.oldaction_dist_logstd: logstds_n}

            thprev = self.gf()

            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                return self.sess.run(self.fvp, feed) + p * config.cg_damping

            g = self.sess.run(self.pg, feed_dict=feed)
            stepdir = conjugate_gradient(fisher_vector_product, -g)
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            assert shs > 0

            lm = np.sqrt(shs / config.max_kl)
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)

            def loss(th):
                self.sff(th)
                return self.sess.run(self.losses[0], feed_dict=feed)
            theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
            self.sff(theta)

            surrafter, kloldnew, entropy = self.sess.run(
                self.losses, feed_dict=feed
            )

            episoderewards = np.array([path["rewards"].sum() for path in paths])
            stats = {}
            numeptotal += len(episoderewards)
            stats["Total number of episodes"] = numeptotal #6570
            stats["Average sum of rewards per episode"] = episoderewards.mean() #760
            stats["Entropy"] = entropy #-16
            stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
            stats["KL between old and new distribution"] = kloldnew ##0.0!!!!!
            stats["Surrogate loss"] = surrafter #-0.40685844
            print("\n********** Iteration {} **********".format(i))
            for k, v in stats.items(): #iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if entropy != entropy:
                print("exit! entropy is ",entropy)
                exit(-1)

            ave_rewards.append(rewards_ave)#自作
            if i%10 == 0 or i<10:
                ######txtで保存
                file2_path = "./bc_nodoor/data_epoch_%d.txt" %i
                f = open(file2_path, "w")
                cc=["loss_p\n","val_loss_p\n","loss_d\n","rewards_ave\n","Time elapsed\n","Surrogate loss\n","Entropy\n","kloldnew\n"]
                aa=[ str(loss_p),"\n" , str(val_loss_p) , "\n" ,str(loss_d), "\n", str(rewards_ave), "\n", str((time.time() - start_time) / 60.0),"\n",str(surrafter),"\n",str(entropy),"\n",str(kloldnew),"\n"]
                f.write("".join(aa))
                f.write("".join(cc))
                f.close()

                param_dir = "./bc_nodoor/"
                print("Now we save model")
                self.generator.save_weights(
                    param_dir + "generator_model_%d.h5" % i, overwrite=True)
                with open(param_dir + "generator_model.json" % i, "w") as outfile:
                    json.dump(self.generator.to_json(), outfile)

                self.discriminator.save_weights(
                    param_dir + "discriminator_model_%d.h5" % i, overwrite=True)
                with open(param_dir + "discriminator_model.json" % i, "w") as outfile:
                    json.dump(self.discriminator.to_json(), outfile)

                self.baseline.model.save_weights(
                    param_dir + "baseline_model_%d.h5" % i, overwrite=True)
                with open(param_dir + "baseline_model.json" % i, "w") as outfile:
                    json.dump(self.baseline.model.to_json(), outfile)

                self.posterior.save_weights(
                    param_dir + "posterior_model_%d.h5" % i, overwrite=True)
                with open(param_dir + "posterior_model.json" % i, "w") as outfile:
                    json.dump(self.posterior.to_json(), outfile)

                self.posterior_target.save_weights(
                    param_dir + "posterior_target_model_%d.h5" % i, overwrite=True)
                with open(param_dir + "posterior_target_model.json" % i, "w") as outfile:
                    json.dump(self.posterior_target.to_json(), outfile)

        x=range(1,n_iter)
        plt.title('Mean returns in each epoch')
        plt.plot(x,ave_rewards,label='classifier')
        plt.xlabel("Epoch")
        plt.ylabel("Mean returns over 10 episode")
        plt.savefig("bc_nodoor/Mean_return.png")


class Detector(object):
    def __init__(self, sess, aux_dim,  encode_dim,action_dim,lr=0.0001):
        self.sess = sess
        #self.lr = tf.placeholder(tf.float32, shape=[])
        K.set_session(sess)
        self.lr=lr
        self.model, self.weights,self.testor, self.planner,self.state_exp, self.action_exp ,self.encodes= \
                self.create_detector( aux_dim, action_dim,encode_dim)
        #self.action_gradient = tf.placeholder(tf.float32, [None, action_dim])
        #self.params_grad = tf.gradients(self.model.output, self.weights, self.action_gradient)
        #grads = zip(self.params_grad, self.weights)
        #self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        ##下自作
        #self.action_exp = tf.placeholder(tf.float32, [None, action_dim])
        #self.state_exp = tf.placeholder(tf.float32, [None, aux_dim])
        """
        self.loss_g = tf.reduce_mean(tf.abs(self.action_exp-self.model_g.output)+tf.abs(self.state_exp-self.model_d.output)) #2乗ならsquare
        self.optimizer_g = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8)
        self.optimize_g = self.optimizer_g.minimize(self.loss_g,var_list=self.weights_g)
        self.loss_d = tf.reduce_mean(tf.abs(self.state_exp-self.model_d.output))
        self.optimizer_d = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8)
        self.optimize_d = self.optimizer_d.minimize(self.loss_d,var_list=self.weights_d)
        """
        self.sess.run(tf.global_variables_initializer())
        print("generator summary")
        #self.model.summary()

    """def train(self, states, actions,  lr):
        self.sess.run([self.loss,self.optimize], feed_dict={
            self.state_exp: states,
            self.action_exp: actions,
            self.lr: lr,
            K.learning_phase(): 1
        })
            #weight_temp = self.model_d.get_weights()
            #self.testor.set_weights(weight_temp[])
        def generate(self,states,)"""

    def create_detector(self,  aux_dim, action_dim, encode_dim):
        states  = Input(shape=[aux_dim])#,name='state_in'
        actions = Input(shape=[action_dim])#,name='action_in'
        input_p = concatenate([states,actions],axis=-1)
        p = Dense(64)(input_p)#256
        p = LeakyReLU()(p)
        p = Dense(64)(p)#128
        p = LeakyReLU()(p)
        c = Dense(encode_dim, activation='softmax')(p)
        #K.function([model.layers[0].input], [model.layers[0].output])
        #layer0_output = layer0_output_fcn([x])[0]
        #print(leyer0_output)
        
        #encoder ここまで
        input_d = concatenate([states,c],axis=-1)
        """
        d1= Dense(64)(input_d)#aux_dim+encode_dim
        d=LeakyReLU()(d1)
        d=Dense(64)(d)#128
        d=LeakyReLU()(d)
        a = Dense(action_dim, activation='tanh')(d)
        decoder_g=Container(inputs=d1,outputs=a)
        """
        decoder_g=Sequential()
        decoder_g.add( Dense(64,input_dim=aux_dim+encode_dim))#aux_dim+encode_dim
        decoder_g.add( LeakyReLU())
        decoder_g.add( Dense(64))#128
        decoder_g.add( LeakyReLU())
        decoder_g.add( Dense(action_dim, activation='tanh'))
        a=decoder_g(input_d)
        #a=decoder_g(np.stack([array_1, array_2], axis=1))
        #decoderここまで
        h = concatenate([states,actions,c],axis=-1)
        decoder_d=Sequential()
        decoder_d.add( Dense(64,input_dim=(aux_dim+action_dim+encode_dim)))#aux_dim+action_dim+encode_dim
        decoder_d.add( LeakyReLU())
        decoder_d.add( Dense(64))#128
        decoder_d.add( LeakyReLU())
        decoder_d.add( Dense(aux_dim, activation='tanh'))
        s=decoder_d(h)
        generator = Model(inputs=[states,actions],outputs=[a,s])

        encodes= Input(shape=[encode_dim])
        #テスト時のdetectorをtestorと命名,c入力
        out_testor=decoder_d(concatenate([states,actions,encodes]))
        testor = Model(inputs=[states,actions,encodes],outputs=out_testor)
        ##テスト時のgenerator,c入力
        out_planner=decoder_g(concatenate([states,encodes]))
        planner = Model(inputs=[states,encodes],outputs=out_planner)
        adam = Adam(lr=self.lr)
        #detector.compile(loss='mean_absolute_error', optimizer=adam,metrics=['accuracy'])
        generator.compile(loss='mean_absolute_error', optimizer=adam,loss_weights=[1,1.0],metrics=['accuracy'])

        return generator, generator.trainable_weights, testor,planner,states, actions ,encodes
class Generator(object):
    def __init__(self, sess,  aux_dim, encode_dim, action_dim):
        self.sess = sess
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.action_dim = action_dim

        K.set_session(sess)

        self.model, self.weights,  self.auxs, self.encodes = \
                self.create_generator( aux_dim, encode_dim)

        self.action_gradient = tf.placeholder(tf.float32, [None, action_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights,
                                        self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        #self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        ##下自作
        self.loss = tf.reduce_mean(tf.square(self.action_gradient-self.model.output))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8)
        self.optimize=self.optimizer.minimize(self.loss,var_list=self.weights)

        self.sess.run(tf.global_variables_initializer())

    def train(self, auxs, encodes, action_grads, lr):
        """
        self.sess.run(self.optimize, feed_dict={
            self.auxs: auxs,
            self.encodes: encodes,
            self.lr: lr,
            self.action_gradient: action_grads,
            K.learning_phase(): 1
        })
        """

        self.sess.run([self.loss,self.optimize], feed_dict={
            self.auxs: auxs,
            self.encodes: encodes,
            self.lr: lr,
            self.action_gradient: action_grads,
            K.learning_phase(): 1
        })

    def create_generator(self, aux_dim, encode_dim):

        auxs = Input(shape=[aux_dim])
        #h = concatenate([x, auxs], axis=-1)
        encodes = Input(shape=[encode_dim])
        h = concatenate([ auxs, encodes],axis=-1)
        h = Dense(64)(h)
        h = LeakyReLU()(h)
        h = Dense(64)(h)
        h = LeakyReLU()(h)
        """
        #TODO
        act1 = Dense(1, activation='tanh')(h)

        act2 = Dense(1, activation='tanh')(h)

        act3 = Dense(1, activation='tanh')(h)

        act4 = Dense(1, activation='tanh')(h)

        act5 = Dense(1, activation='tanh')(h)

        act6 = Dense(1, activation='tanh')(h)

        actions = concatenate([act1,act2,act3,act4,act5,act6],axis=-1)"""
        actions = Dense(self.action_dim, activation='tanh')(h)
        #actions = merge([act1,act2,act3,act4,act5,act6], mode='concat')
        model = Model(inputs=[ auxs, encodes], outputs=actions)
        return model, model.trainable_weights, auxs, encodes


class Posterior(object):
    def __init__(self, sess,  aux_dim, action_dim, encode_dim):
        self.sess = sess
        self.lr = tf.placeholder(tf.float32, shape=[])

        K.set_session(sess)

        self.model = self.create_posterior( aux_dim, action_dim, encode_dim)

    def create_posterior(self, aux_dim, action_dim, encode_dim):

        auxs = Input(shape=[aux_dim])
        actions = Input(shape=[action_dim])
        #h = merge([x, auxs, actions], mode='concat')
        h = concatenate([auxs, actions],axis=-1)
        h = Dense(64)(h) #256
        h = LeakyReLU()(h)
        h = Dense(64)(h) #128
        h = LeakyReLU()(h)
        c = Dense(encode_dim, activation='softmax')(h)

        model = Model(inputs=[auxs, actions], outputs=c)
        return model
