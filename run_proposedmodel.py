# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import json
import time
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from propmodels import Detector
#from models2 import Generator
import os
import logging
import warnings
import matplotlib
matplotlib.use("Agg") ##GUIない環境では必要
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

def calc_loss(act_gt, act_pred):##absをsquareにした自作
    steer_loss = np.average(np.abs(act_gt[:, 0] - act_pred[:, 0]))
    accel_loss = np.average(np.abs(act_gt[:, 1] - act_pred[:, 1]))
    brake_loss = np.average(np.abs(act_gt[:, 2] - act_pred[:, 2]))
    loss3 = np.average(np.abs(act_gt[:, 3] - act_pred[:, 3]))
    loss4 = np.average(np.abs(act_gt[:, 4] - act_pred[:, 4]))
    loss5 = np.average(np.abs(act_gt[:, 5] - act_pred[:, 5]))
    return np.array([steer_loss, accel_loss, brake_loss,loss3,loss4,loss5])

def main():
    param_dir = "./proposed_model/"
    os.makedirs(param_dir, exist_ok=True)
    #pre_actions_path = "./proposed_model/demo_nondoor_40.npz"
    demo_dir = "./proposed_model/demo_RGB_500.npz"#demo_rgba_push.npz" #"./bc_nocheetah/Halfcheetah_traj_slow.npz" #
    aux_dim = 36#15
    encode_dim = 1
    action_dim = 6
    batch_size = 128#64#512#demo.shape0は1600より大きい? TODOデモセット数*train ratio
    epoch=100#500

    train_val_ratio = 0.8
    lr = 0.0001
    lr_decay_factor = .99
    #episode = 4001#501 #train no kurikaesu kaisuu
    np.random.seed(1024)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from tensorflow.keras import backend as K
    K.set_session(sess)

    print ("Now we build generator")
    generator = Detector(sess, aux_dim, encode_dim, action_dim,lr)
    #generator.model.load_weights(param_dir + "generator_bc_model_push2500.h5")

    print ("Loading data ...")
    #TODO
    raw = np.load(demo_dir)
    auxs, actions =  raw["obs"], raw["actions"]
    num_epi = int(auxs.shape[0]/512)
    saigo=[i*512+511 for i in range(num_epi)]
    saisho=[i*512 for i in range(num_epi)]
    #標準化
    auxs_mean=np.mean(auxs,axis=0, keepdims=True)
    auxs_std=np.std(auxs,axis=0, keepdims=True)
    #auxs=(auxs-auxs_mean)/auxs_std
    #print("asafds",auxs.shape)
    file2_path = param_dir+"mean,std.txt"
    f = open(file2_path, "w")
    f.write("Aux Mean "+str(auxs_mean)+"\n")
    f.write("Aux Std "+str(auxs_std))
    f.close()


    auxs_next     = np.delete(auxs,saisho,0)
    actions_next  = np.delete(actions,saigo,0)
    auxs_train    = np.delete(auxs,saigo,0)
    actions_train = np.delete(actions,saigo,0)

    #test用
    auxs_l=auxs_next[:511]
    actions_l=actions_next[:511]
    auxs_t=auxs_train[:511]
    actions_t=actions_train[:511]
    """###episode単位でシャッフルしたほうがいい
    actions = actions.reshape(-1,512,action_dim)
    auxs = auxs.reshape(-1,512,aux_dim)"""
    """
    num_data = auxs.shape[0]
    idx = np.arange(num_data)
    np.random.shuffle(idx)"""
    #auxs=auxs.reshape(-1,aux_dim)
    result = generator.model.fit([auxs_train,actions_train],[actions_next,auxs_next],epochs=epoch,batch_size=batch_size,validation_data=([auxs_t,actions_t],[actions_l,auxs_l]))

    print("Now we save the model")
    generator.model.save_weights(param_dir + "generator_{}.h5".format(epoch), overwrite=True)
    with open(param_dir + "generator.json", "w") as outfile:
        json.dump(generator.model.to_json(), outfile)
    generator.planner.save_weights(param_dir + "planner_{}.h5".format(epoch), overwrite=True)
    with open(param_dir + "planner.json", "w") as outfile:
        json.dump(generator.planner.to_json(), outfile)
    generator.testor.save_weights(param_dir + "testor_{}.h5".format(epoch), overwrite=True)
    with open(param_dir + "testor.json", "w") as outfile:
        json.dump(generator.testor.to_json(), outfile)

    #print("aasa\n",result.history.keys()) dict_keys(['loss', 'sequential_loss', 'sequential_1_loss', 'sequential_acc', 'sequential_1_acc', 'val_loss', 'val_sequential_loss', 'val_sequential_1_loss', 'val_sequential_acc', 'val_sequential_1_acc'])

    plt.plot(range(1, epoch+1), result.history['loss'], label="training")
    plt.plot(range(1, epoch+1), result.history['val_loss'], label="validation")
    plt.ylim([0,4])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(param_dir+"loss_epoch:{}_batch_{}.png".format(epoch,batch_size))
    """
    print("original data",auxs.shape,actions.shape)#(10, 512, 50, 50, 3), (10, 512, 19), (10, 512, 6)
    auxs_train = auxs[idx][:int(num_data * train_val_ratio)]
    actions_train = actions[idx][:int(num_data * train_val_ratio)]
    auxs_val = auxs[idx][int(num_data * train_val_ratio):]
    actions_val = actions[idx][int(num_data * train_val_ratio):]
    print ("Getting feature for training set ...")

    #print("action",actions_val.shape) #(2,512,6)
    actions_train = actions_train.reshape(-1,action_dim)
    actions_val = actions_val.reshape(-1,action_dim)
    auxs_train = auxs_train.reshape(-1,aux_dim)
    auxs_val = auxs_val.reshape(-1,aux_dim)"""

    cur_min_loss_val = 2.6
    trainloss= []
    testloss = []
    ###途中から学習したい。
    #generator.load_weights(param_dir + "generator_bc_model_500.h5")



    """
    for i in range(episode):
        total_step = auxs_train.shape[0] // batch_size
        #print("adsfadsfa",total_step)#8

        train_loss = np.array([0., 0., 0., 0., 0., 0.])#TODO action_dimに合わせる
        for j in range(total_step):
            auxs_cur = auxs_train[j * batch_size : (j + 1) * batch_size]
            encodes_cur = np.zeros([batch_size, encode_dim], dtype=np.float32)
            idx = np.random.randint(0, encode_dim, batch_size)
            encodes_cur[np.arange(batch_size), idx] = 1#encodeランダムにしてる
            #print("encodes_cur",encodes_cur.shape)#[512,1] 2次元

            act_pred = generator.model.predict([auxs_cur,encodes_cur])
            act_gt = actions_train[j * batch_size : (j + 1) * batch_size]
            #print(act_pred.shape,"adsf")#64,6


            generator.train(auxs_cur, encodes_cur, act_gt, lr)#act_pred - act_gt自作
            batch_loss = calc_loss(act_gt, act_pred)
            #print ("Episode:", i, "Batch:", j, "/", total_step,np.round(batch_loss, 6), np.sum(batch_loss) )

            train_loss += batch_loss / total_step

        if i % 20 == 0 and i > 0:
            lr *= lr_decay_factor

        encodes_val = np.zeros([auxs_val.shape[0], encode_dim], dtype=np.float32)
        idx = np.random.randint(0, encode_dim, auxs_val.shape[0])
        encodes_val[idx] = 1
        #print(feats_val.shape, auxs_val.shape, encodes_val.shape)#(1024, 3, 3, 1024), (1024, 19), (1024, 2)

        act_pred = generator.model.predict([ auxs_val, encodes_val])
        val_loss = calc_loss(actions_val, act_pred)
        print ("Episode:", i, \
            "Train Loss: ", np.round(train_loss, 6), np.sum(train_loss), \
            "Test Loss:", np.round(val_loss, 6), np.sum(val_loss), cur_min_loss_val, \
            "LR:", lr)

        #lossの記録
        trainloss.append(np.sum(train_loss).tolist())
        testloss.append(np.sum(val_loss).tolist())


        #if cur_min_loss_val > np.sum(val_loss):
        if cur_min_loss_val > np.sum(val_loss):
            cur_min_loss_val = np.sum(val_loss)
            print("Now we save the model")
            generator.model.save_weights(param_dir + "generator_bc_model36_{}.h5".format(i), overwrite=True)
            with open(param_dir + "generator_bc_model.json", "w") as outfile:
                json.dump(generator.model.to_json(), outfile)

        if i%500 == 0:
            print("Now we save the model")
            generator.model.save_weights(param_dir + "generator_bc_model36_{}.h5".format(i), overwrite=True)
            with open(param_dir + "generator_bc_model.json", "w") as outfile:
                json.dump(generator.model.to_json(), outfile)

    """
    ##tensorboard --logdir=logs/する用
    #summary_writer = tf.summary.FileWriter('logs', sess.graph)
    #summary_writer.close()
    ####pyplot
    """
    plt.plot(testloss,label='testloss')#2000epochで4.9071
    plt.plot(trainloss,label='trainloss')#2000epochで7.6
    plt.legend()
    plt.savefig(param_dir+"bc36.png")
    ######txtで保存
    file2_path = param_dir+"data36_%d.txt" %i
    f = open(file2_path, "w")
    f.write("Train Loss "+str(train_loss)+"\n")
    f.write("Test Loss "+str(val_loss))
    f.close()"""



if __name__ == "__main__":
    main()
