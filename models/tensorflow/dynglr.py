import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from models.tensorflow.utils_data import *
from models.tensorflow.utils_gpu import *
from models.tensorflow.utils_arch import *
from tqdm import tqdm
from utils import *

class DYNGLR:
    def __init__(self, classes, **kwargs):
        set_session(**kwargs)
        tf.random.set_seed(kwargs.get('randstate', 123))
        self.classes = classes
        
        self.compute_loss = TripletGLRLossError(margin=kwargs.get('dynglr_margin', 5.0))

        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.optimizer1 = tf.keras.optimizers.Adam(0.001)
        self.optimizer2 = tf.keras.optimizers.Adam(0.001)

        self.lr_mode_stages = kwargs.get('lr_mode_stages', None)
        self.lrs_stages= kwargs.get('lrs_stages', None)
        self.lr_stairs_stages = kwargs.get('lr_stairs_stages', None)
        self.lr_init_stages = kwargs.get('lr_init_stages', None)
        self.lr_vals_stages = kwargs.get('lr_vals_stages', None)
        self.lr_knots_stages = kwargs.get('lr_knots_stages', None)

        if is_sequence(self.lr_mode_stages):
            lr_init = self.lr_init_stages[0] if is_sequence(self.lr_init_stages) else self.lr_init_stages
            lrs = self.lrs_stages[0] if is_sequence(self.lrs_stages) else self.lrs_stages
            lr_stairs = self.lr_stairs_stages[0] if is_sequence(self.lr_stairs_stages) else self.lr_stairs_stages
            lr_vals = self.lr_vals_stages[0] if is_sequence(self.lr_vals_stages) else self.lr_vals_stages
            lr_knots = self.lr_knots_stages[0] if is_sequence(self.lr_knots_stages) else self.lr_knots_stages
            self.lr = get_arch_lr(lr_mode=self.lr_mode_stages[0], lr_init=lr_init, lrs=lrs, lr_stairs=lr_stairs, lr_vals=lr_vals, lr_knots=lr_knots)

            lr_init = self.lr_init_stages[1] if is_sequence(self.lr_init_stages) else self.lr_init_stages
            lrs = self.lrs_stages[1] if is_sequence(self.lrs_stages) else self.lrs_stages
            lr_stairs = self.lr_stairs_stages[1] if is_sequence(self.lr_stairs_stages) else self.lr_stairs_stages
            lr_vals = self.lr_vals_stages[1] if is_sequence(self.lr_vals_stages) else self.lr_vals_stages
            lr_knots = self.lr_knots_stages[1] if is_sequence(self.lr_knots_stages) else self.lr_knots_stages
            self.lr1 = get_arch_lr(lr_mode=self.lr_mode_stages[1], lr_init=lr_init, lrs=lrs, lr_stairs=lr_stairs, lr_vals=lr_vals, lr_knots=lr_knots)

            lr_init = self.lr_init_stages[2] if is_sequence(self.lr_init_stages) else self.lr_init_stages
            lrs = self.lrs_stages[2] if is_sequence(self.lrs_stages) else self.lrs_stages
            lr_stairs = self.lr_stairs_stages[2] if is_sequence(self.lr_stairs_stages) else self.lr_stairs_stages
            lr_vals = self.lr_vals_stages[2] if is_sequence(self.lr_vals_stages) else self.lr_vals_stages
            lr_knots = self.lr_knots_stages[2] if is_sequence(self.lr_knots_stages) else self.lr_knots_stages
            self.lr2 = get_arch_lr(lr_mode=self.lr_mode_stages[2], lr_init=lr_init, lrs=lrs, lr_stairs=lr_stairs, lr_vals=lr_vals, lr_knots=lr_knots)
        else:
            self.lr = get_arch_lr(lr_mode=self.lr_mode_stages, lr_init=self.lr_init_stages, lrs=self.lr_init_stages, lr_stairs=self.lr_stairs_stages, lr_vals=self.lr_vals_stages, lr_knots=self.lr_knots_stages)
            self.lr1 = get_arch_lr(lr_mode=self.lr_mode_stages, lr_init=self.lr_init_stages, lrs=self.lr_init_stages, lr_stairs=self.lr_stairs_stages, lr_vals=self.lr_vals_stages, lr_knots=self.lr_knots_stages)
            self.lr2 = get_arch_lr(lr_mode=self.lr_mode_stages, lr_init=self.lr_init_stages, lrs=self.lr_init_stages, lr_stairs=self.lr_stairs_stages, lr_vals=self.lr_vals_stages, lr_knots=self.lr_knots_stages)

        self.step = 0
        self.step1 = 0
        self.step2 = 0

        self.metrics = {
            'wnet1_train_loss': TripletGLRLoss(),
            'wnet1_train_refloss': TripletGLRLoss(),
            'hnet_train_loss': TripletGLRLoss(),
            'hnet_train_refloss': TripletGLRLoss(),
            'wnet2_train_loss': TripletGLRLoss(),
            'wnet1_valid_acc': tf.keras.metrics.Accuracy(),
            'wnet1_valid_refacc': tf.keras.metrics.Accuracy(),
            'hnet_valid_loss': TripletGLRLoss(),
            'hnet_valid_refloss': TripletGLRLoss(),
            'wnet2_valid_acc': tf.keras.metrics.Accuracy(),
            'wnet2_valid_refacc': tf.keras.metrics.Accuracy(),
            'g2_acc': tf.keras.metrics.Accuracy(),
            'g12_acc': tf.keras.metrics.Accuracy(),
            'g1232_acc': tf.keras.metrics.Accuracy(),
            'g12312_acc': tf.keras.metrics.Accuracy(),
            }

        self.network_w1, self.network_u, self.network_w2 = tf_arch(self.classes, **kwargs)

    @tf.function
    def train_wnet1(self, lx, ly, lry, x, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])

        with tf.GradientTape() as tape:
            batchsize = tf.shape(x)[0]
            labelednum = tf.shape(lx)[1]

            ax = tf.concat((lx,x),1)
            nodenum = tf.shape(ax)[1]
            unlabelednum = nodenum - labelednum

            latents, _ = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), True)
            latents = tf.reshape(latents, [batchsize, nodenum, -1])
            dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
            labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

            dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
            edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

            sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
            w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
            adj = tf.multiply(w, edge)

            glr_ret, glr_gs, glr_sgs = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

            nm, m = edgeattention(glr_gs, glr_sgs, labelednum, unlabelednum, kwargs.get('dynglr_thres', 2.0))

            loss_m = tf.slice(m, [0,0,0], [batchsize, labelednum, labelednum])
            loss = self.compute_loss(ly, labeled_dists, loss_m)

            self.metrics['wnet1_train_loss'].update_state(ly, labeled_dists, loss_m)
            self.metrics['wnet1_train_refloss'].update_state(lry, labeled_dists, loss_m)

        grads = tape.gradient(loss, self.network_w1.trainable_variables)
        grads = postprocess_grads(grads, **kwargs)
        self.optimizer.learning_rate = self.lr(self.step)
        self.optimizer.apply_gradients(zip(grads, self.network_w1.trainable_variables))
        self.step+=1

    @tf.function
    def train_hnet(self, lx, ly, lry, x, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])

        with tf.GradientTape() as tape:
            batchsize = tf.shape(x)[0]
            labelednum = tf.shape(lx)[1]

            ax = tf.concat((lx,x),1)
            nodenum = tf.shape(ax)[1]
            unlabelednum = nodenum - labelednum

            #wnet1
            latents, shallows = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), False)
            latents = tf.reshape(latents, [batchsize, nodenum, -1])
            dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
            labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

            dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
            edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

            sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
            w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
            adj = tf.multiply(w, edge)

            glr_ret, glr_gs, glr_sgs = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

            nm, m = edgeattention(glr_gs, glr_sgs, labelednum, unlabelednum, kwargs.get('dynglr_thres', 2.0))

            denoise_ly = tf.cast(tf.greater(tf.slice(glr_ret, [0,0], [batchsize, labelednum]), 0.0), tf.int32)

            loss_m = tf.slice(m, [0,0,0], [batchsize, labelednum, labelednum])

            #hnet
            glr_sign_ret = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
            ency = tf.one_hot(glr_sign_ret, 2) *  tf.math.abs(tf.expand_dims(glr_ret, 2))
            mau = tf.abs(tf.slice(glr_gs, [0,0], [batchsize, labelednum]) -  tf.slice(glr_sgs, [0,0], [batchsize, labelednum]))
            mau_mask = tf.cast(tf.greater(mau, kwargs.get('dynglr_thres', 2.0)), tf.float32)
            mau = tf.cast(tf.multiply(1.0 + tf.multiply(tf.reduce_max(mau, 1, keepdims=True) - mau, (kwargs.get('dynglr_tau', 3)-1)), mau_mask), tf.int32)
            mau = tf.concat((mau, tf.zeros((batchsize, unlabelednum), tf.int32)), 1)
            mau = tf.ones_like(mau, tf.int32)*kwargs.get('dynglr_tau', 3)

            dency = dynglr_knnfeature(ency, w, kwargs.get('dynglr_tau', 3), mau)
            dency = tf.reshape(dency, [batchsize * nodenum, -1])

            latents1, shallows1 = self.network_u([shallows, dency], True)
            latents1 = tf.reshape(latents1, [batchsize, nodenum, -1])
            dists1 = euclidean_distance(latents1, kwargs.get('use_fp16', False))
            labeled_dists1 = tf.slice(dists1, [0, 0,0], [batchsize, labelednum, labelednum])

            loss = self.compute_loss(denoise_ly, labeled_dists1, loss_m)

            self.metrics['hnet_train_loss'].update_state(denoise_ly, labeled_dists1, loss_m)
            self.metrics['hnet_train_refloss'].update_state(lry, labeled_dists1, loss_m)

        grads = tape.gradient(loss, self.network_u.trainable_variables)
        grads = postprocess_grads(grads, **kwargs)
        self.optimizer1.learning_rate = self.lr1(self.step1)
        self.optimizer1.apply_gradients(zip(grads, self.network_u.trainable_variables))
        self.step1+=1

    @tf.function
    def train_wnet2(self, lx, ly, lry, x, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])

        with tf.GradientTape() as tape:
            batchsize = tf.shape(x)[0]
            labelednum = tf.shape(lx)[1]

            ax = tf.concat((lx,x),1)
            nodenum = tf.shape(ax)[1]
            unlabelednum = nodenum - labelednum

            #wnet1
            latents, shallows = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), False)
            latents = tf.reshape(latents, [batchsize, nodenum, -1])
            dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
            labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

            dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
            edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

            sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
            w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
            adj = tf.multiply(w, edge)

            glr_ret, glr_gs, glr_sgs = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

            denoise_ay = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
            denoise_ly = tf.slice(denoise_ay, [0,0], [batchsize, labelednum])

            #hnet
            glr_sign_ret = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
            ency = tf.one_hot(glr_sign_ret, 2) *  tf.math.abs(tf.expand_dims(glr_ret, 2))
            mau = tf.abs(tf.slice(glr_gs, [0,0], [batchsize, labelednum]) -  tf.slice(glr_sgs, [0,0], [batchsize, labelednum]))
            mau_mask = tf.cast(tf.greater(mau, kwargs.get('dynglr_thres', 2.0)), tf.float32)
            mau = tf.cast(tf.multiply(1.0 + tf.multiply(tf.reduce_max(mau, 1, keepdims=True) - mau, (kwargs.get('dynglr_tau', 3)-1)), mau_mask), tf.int32)
            mau = tf.concat((mau, tf.zeros((batchsize, unlabelednum), tf.int32)), 1)
            mau = tf.ones_like(mau, tf.int32)*kwargs.get('dynglr_tau', 3)

            dency = dynglr_knnfeature(ency, w, kwargs.get('dynglr_tau', 3), mau)
            dency = tf.reshape(dency, [batchsize * nodenum, -1])

            latents1, shallows1 = self.network_u([shallows, dency], False)
            latents1 = tf.reshape(latents1, [batchsize, nodenum, -1])
            dists1 = euclidean_distance(latents1, kwargs.get('use_fp16', False))
            labeled_dists1 = tf.slice(dists1, [0, 0,0], [batchsize, labelednum, labelednum])

            sparse_gamma = tf.cast(tf.greater(adj, 0.1), tf.float32)
            sparse_gamma = tf.cast(tf.reduce_sum(sparse_gamma, 2), tf.int32)
            edge1 = sparse_knn(dists1, sparse_gamma, nodenum)

            #wnet2
            latents2 = self.network_w2([shallows, shallows1], True)
            latents2 = tf.reshape(latents2, [batchsize, nodenum, -1])
            dists2 = euclidean_distance(latents2, kwargs.get('use_fp16', False))
            labeled_dists2 = tf.slice(dists2, [0, 0,0], [batchsize, labelednum, labelednum])

            sigma2 = adaptive_sigma(labeled_dists2, denoise_ly, use_fp16=kwargs.get('use_fp16', False))
            w2,_ = edge_weighting(dists2, sigma2, use_fp16=kwargs.get('use_fp16', False))
            adj2 = tf.multiply(w2, edge1)

            glr_ret2, glr_gs2, glr_sgs2 = glr_fidelity(adj2, glr_ret, kwargs.get('dynglr_mu2', 1.0), labelednum, unlabelednum, nodenum, ally=True, normlap=kwargs.get('dynglr_norm', False))

            nm, m = edgeattention(glr_gs2, glr_sgs2, labelednum, unlabelednum, kwargs.get('dynglr_thres2', 2.0))

            loss_m = m
            loss = self.compute_loss(denoise_ay, dists2, loss_m)
            self.metrics['wnet2_train_loss'].update_state(denoise_ay, dists2, loss_m)

        grads = tape.gradient(loss, self.network_w2.trainable_variables)
        grads = postprocess_grads(grads, **kwargs)
        self.optimizer2.learning_rate = self.lr2(self.step2)
        self.optimizer2.apply_gradients(zip(grads, self.network_w2.trainable_variables))
        self.step2+=1

    @tf.function
    def validate_wnet1(self, lx, ly, lry, x, y, ry, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])

        batchsize = tf.shape(x)[0]
        labelednum = tf.shape(lx)[1]

        ax = tf.concat((lx,x),1)
        nodenum = tf.shape(ax)[1]
        unlabelednum = nodenum - labelednum

        latents, _ = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), False)
        latents = tf.reshape(latents, [batchsize, nodenum, -1])
        dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
        labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

        dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
        edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

        sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
        w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
        adj = tf.multiply(w, edge)

        glr_ret, _, _ = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

        glr_sret = tf.slice(glr_ret, [0,labelednum], [batchsize, unlabelednum])
        ret = tf.cast(tf.greater(glr_sret, 0.0), tf.int32)

        if avg_eval:
            ret = tf.cast(tf.greater_equal(tf.reduce_sum(ret,0, keepdims=True), np.ceil(len(lx)*0.5)), tf.int32)

        self.metrics['wnet1_valid_acc'].update_state(y, ret)
        self.metrics['wnet1_valid_refacc'].update_state(ry, ret)

    @tf.function
    def validate_hnet(self, lx, ly, lry, x, y, ry, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])

        batchsize = tf.shape(x)[0]
        labelednum = tf.shape(lx)[1]

        ax = tf.concat((lx,x),1)
        nodenum = tf.shape(ax)[1]
        unlabelednum = nodenum - labelednum

        latents, shallows = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), False)
        latents = tf.reshape(latents, [batchsize, nodenum, -1])
        dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
        labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

        dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
        edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

        sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
        w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
        adj = tf.multiply(w, edge)

        glr_ret, glr_gs, glr_sgs = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

        nm, m = edgeattention(glr_gs, glr_sgs, labelednum, unlabelednum, kwargs.get('dynglr_thres', 2.0))

        denoise_ly = tf.cast(tf.greater(tf.slice(glr_ret, [0,0], [batchsize, labelednum]), 0.0), tf.int32)

        loss_m = tf.slice(m, [0,0,0], [batchsize, labelednum, labelednum])

        #hnet
        glr_sign_ret = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
        ency = tf.one_hot(glr_sign_ret, 2) *  tf.math.abs(tf.expand_dims(glr_ret, 2))
        mau = tf.abs(tf.slice(glr_gs, [0,0], [batchsize, labelednum]) -  tf.slice(glr_sgs, [0,0], [batchsize, labelednum]))
        mau_mask = tf.cast(tf.greater(mau, kwargs.get('dynglr_thres', 2.0)), tf.float32)
        mau = tf.cast(tf.multiply(1.0 + tf.multiply(tf.reduce_max(mau, 1, keepdims=True) - mau, (kwargs.get('dynglr_tau', 3)-1)), mau_mask), tf.int32)
        mau = tf.concat((mau, tf.zeros((batchsize, unlabelednum), tf.int32)), 1)
        mau = tf.ones_like(mau, tf.int32)*kwargs.get('dynglr_tau', 3)

        dency = dynglr_knnfeature(ency, w, kwargs.get('dynglr_tau', 3), mau)
        dency = tf.reshape(dency, [batchsize * nodenum, -1])

        latents1, shallows1 = self.network_u([shallows, dency], False)
        latents1 = tf.reshape(latents1, [batchsize, nodenum, -1])
        dists1 = euclidean_distance(latents1, kwargs.get('use_fp16', False))
        labeled_dists1 = tf.slice(dists1, [0, 0,0], [batchsize, labelednum, labelednum])

        self.metrics['hnet_valid_loss'].update_state(denoise_ly, labeled_dists1, loss_m)
        self.metrics['hnet_valid_refloss'].update_state(lry, labeled_dists1, loss_m)

    @tf.function
    def validate_wnet2(self, lx, ly, lry, x, y, ry, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])

        batchsize = tf.shape(x)[0]
        labelednum = tf.shape(lx)[1]

        ax = tf.concat((lx,x),1)
        nodenum = tf.shape(ax)[1]
        unlabelednum = nodenum - labelednum

        #wnet1
        latents, shallows = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), False)
        latents = tf.reshape(latents, [batchsize, nodenum, -1])
        dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
        labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

        dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
        edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

        sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
        w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
        adj = tf.multiply(w, edge)

        glr_ret, glr_gs, glr_sgs = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

        denoise_ay = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
        denoise_ly = tf.slice(denoise_ay, [0,0], [batchsize, labelednum])

        #hnet
        glr_sign_ret = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
        ency = tf.one_hot(glr_sign_ret, 2) *  tf.math.abs(tf.expand_dims(glr_ret, 2))
        mau = tf.abs(tf.slice(glr_gs, [0,0], [batchsize, labelednum]) -  tf.slice(glr_sgs, [0,0], [batchsize, labelednum]))
        mau_mask = tf.cast(tf.greater(mau, kwargs.get('dynglr_thres', 2.0)), tf.float32)
        mau = tf.cast(tf.multiply(1.0 + tf.multiply(tf.reduce_max(mau, 1, keepdims=True) - mau, (kwargs.get('dynglr_tau', 3)-1)), mau_mask), tf.int32)
        mau = tf.concat((mau, tf.zeros((batchsize, unlabelednum), tf.int32)), 1)
        mau = tf.ones_like(mau, tf.int32)*kwargs.get('dynglr_tau', 3)

        dency = dynglr_knnfeature(ency, w, kwargs.get('dynglr_tau', 3), mau)
        dency = tf.reshape(dency, [batchsize * nodenum, -1])

        latents1, shallows1 = self.network_u([shallows, dency], False)
        latents1 = tf.reshape(latents1, [batchsize, nodenum, -1])
        dists1 = euclidean_distance(latents1, kwargs.get('use_fp16', False))
        labeled_dists1 = tf.slice(dists1, [0, 0,0], [batchsize, labelednum, labelednum])

        sparse_gamma = tf.cast(tf.greater(adj, 0.1), tf.float32)
        sparse_gamma = tf.cast(tf.reduce_sum(sparse_gamma, 2), tf.int32)
        edge1 = sparse_knn(dists1, sparse_gamma, nodenum)

        #wnet2
        latents2 = self.network_w2([shallows, shallows1], False)
        latents2 = tf.reshape(latents2, [batchsize, nodenum, -1])
        dists2 = euclidean_distance(latents2, kwargs.get('use_fp16', False))
        labeled_dists2 = tf.slice(dists2, [0, 0,0], [batchsize, labelednum, labelednum])

        sigma2 = adaptive_sigma(labeled_dists2, denoise_ly, use_fp16=kwargs.get('use_fp16', False))
        w2,_ = edge_weighting(dists2, sigma2, use_fp16=kwargs.get('use_fp16', False))
        adj2 = tf.multiply(w2, edge1)

        glr_ret2, _, _ = glr_fidelity(adj2, glr_ret, kwargs.get('dynglr_mu2', 1.0), labelednum, unlabelednum, nodenum, ally=True, normlap=kwargs.get('dynglr_norm', False), kappafactor=kwargs.get('kappa_factor', 0.5))

        glr_sret = tf.slice(glr_ret2, [0,labelednum], [batchsize, unlabelednum])
        ret = tf.cast(tf.greater(glr_sret, 0.0), tf.int32)

        if avg_eval:
            ret = tf.cast(tf.greater_equal(tf.reduce_sum(ret,0, keepdims=True), np.ceil(len(lx)*0.5)), tf.int32)

        self.metrics['wnet2_valid_acc'].update_state(y, ret)
        self.metrics['wnet2_valid_refacc'].update_state(ry, ret)

    @tf.function
    def predict(self, lx, ly, x, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])

        batchsize = tf.shape(x)[0]
        labelednum = tf.shape(lx)[1]

        ax = tf.concat((lx,x),1)
        nodenum = tf.shape(ax)[1]
        unlabelednum = nodenum - labelednum

        #wnet1
        latents, shallows = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), False)
        latents = tf.reshape(latents, [batchsize, nodenum, -1])
        dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
        labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

        dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
        edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

        sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
        w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
        adj = tf.multiply(w, edge)

        glr_ret, glr_gs, glr_sgs = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

        denoise_ay = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
        denoise_ly = tf.slice(denoise_ay, [0,0], [batchsize, labelednum])

        #hnet
        glr_sign_ret = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
        ency = tf.one_hot(glr_sign_ret, 2) *  tf.math.abs(tf.expand_dims(glr_ret, 2))
        mau = tf.abs(tf.slice(glr_gs, [0,0], [batchsize, labelednum]) -  tf.slice(glr_sgs, [0,0], [batchsize, labelednum]))
        mau_mask = tf.cast(tf.greater(mau, kwargs.get('dynglr_thres', 2.0)), tf.float32)
        mau = tf.cast(tf.multiply(1.0 + tf.multiply(tf.reduce_max(mau, 1, keepdims=True) - mau, (kwargs.get('dynglr_tau', 3)-1)), mau_mask), tf.int32)
        mau = tf.concat((mau, tf.zeros((batchsize, unlabelednum), tf.int32)), 1)
        mau = tf.ones_like(mau, tf.int32)*kwargs.get('dynglr_tau', 3)

        dency = dynglr_knnfeature(ency, w, kwargs.get('dynglr_tau', 3), mau)
        dency = tf.reshape(dency, [batchsize * nodenum, -1])

        latents1, shallows1 = self.network_u([shallows, dency], False)
        latents1 = tf.reshape(latents1, [batchsize, nodenum, -1])
        dists1 = euclidean_distance(latents1, kwargs.get('use_fp16', False))
        labeled_dists1 = tf.slice(dists1, [0, 0,0], [batchsize, labelednum, labelednum])

        sparse_gamma = tf.cast(tf.greater(adj, 0.1), tf.float32)
        sparse_gamma = tf.cast(tf.reduce_sum(sparse_gamma, 2), tf.int32)
        edge1 = sparse_knn(dists1, sparse_gamma, nodenum)

        #wnet2
        latents2 = self.network_w2([shallows, shallows1], False)
        latents2 = tf.reshape(latents2, [batchsize, nodenum, -1])
        dists2 = euclidean_distance(latents2, kwargs.get('use_fp16', False))
        labeled_dists2 = tf.slice(dists2, [0, 0,0], [batchsize, labelednum, labelednum])

        sigma2 = adaptive_sigma(labeled_dists2, denoise_ly, use_fp16=kwargs.get('use_fp16', False))
        w2,_ = edge_weighting(dists2, sigma2, use_fp16=kwargs.get('use_fp16', False))
        adj2 = tf.multiply(w2, edge1)

        glr_ret2, _, _ = glr_fidelity(adj2, glr_ret, kwargs.get('dynglr_mu2', 1.0), labelednum, unlabelednum, nodenum, ally=True, normlap=kwargs.get('dynglr_norm', False), kappafactor=kwargs.get('kappa_factor', 0.5))

        glr_sret = tf.slice(glr_ret2, [0,labelednum], [batchsize, unlabelednum])
        ret = tf.cast(tf.greater(glr_sret, 0.0), tf.int32)

        if avg_eval:
            ret = tf.cast(tf.greater_equal(tf.reduce_sum(ret,0, keepdims=True), np.ceil(len(lx)*0.5)), tf.int32)

        return ret

    @tf.function
    def evaluate_g2(self, lx, ly, lry, x, y, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])
            my = tf.tile(y, [len(ly), 1])
            ay = tf.concat((ly, my), 1)
            ray = tf.concat((lry, my), 1)
            ag = binarize_gs(tf.concat((ly, tf.tile(tf.zeros_like(y), [len(ly), 1])), 1))
        else:
            my = y
            ay = tf.concat((ly, y), 1)
            ray = tf.concat((lry, y), 1)
            ag = binarize_gs(tf.concat((ly, y), 1))

        batchsize = tf.shape(x)[0]
        labelednum = tf.shape(lx)[1]

        ax = tf.concat((lx,x),1)
        nodenum = tf.shape(ax)[1]
        unlabelednum = nodenum - labelednum

        dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
        edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))
        adj = edge

        glr_ret, _, _ = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('g2_normlap', False))

        glr_sret = tf.slice(glr_ret, [0,labelednum], [batchsize, unlabelednum])
        #ret = tf.cast(tf.greater(glr_sret, 0.0), tf.int32)
        ret = glr_sret

        #update acc
        if avg_eval:
            #ret = tf.cast(tf.greater_equal(tf.reduce_sum(ret,0, keepdims=True), np.ceil(len(lx)*0.5)), tf.int32)
            ret = tf.cast(tf.greater(tf.reduce_sum(ret,0, keepdims=True), 0.0), tf.int32)
            self.metrics['g2_acc'].update_state(y, ret)
        else:
            self.metrics['g2_acc'].update_state(my, ret)
    
    @tf.function
    def evaluate_g12(self, lx, ly, lry, x, y, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])
            my = tf.tile(y, [len(ly), 1])
            ay = tf.concat((ly, my), 1)
            ray = tf.concat((lry, my), 1)
            ag = binarize_gs(tf.concat((ly, tf.tile(tf.zeros_like(y), [len(ly), 1])), 1))
        else:
            my = y
            ay = tf.concat((ly, y), 1)
            ray = tf.concat((lry, y), 1)
            ag = binarize_gs(tf.concat((ly, y), 1))

        batchsize = tf.shape(x)[0]
        labelednum = tf.shape(lx)[1]

        ax = tf.concat((lx,x),1)
        nodenum = tf.shape(ax)[1]
        unlabelednum = nodenum - labelednum

        dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
        edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

        #wnet1
        latents, shallows = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), False)
        latents = tf.reshape(latents, [batchsize, nodenum, -1])
        dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
        labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

        sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
        w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
        adj = tf.multiply(w, edge)

        glr_ret, _, _ = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

        glr_sret = tf.slice(glr_ret, [0,labelednum], [batchsize, unlabelednum])
        #ret = tf.cast(tf.greater(glr_sret, 0.0), tf.int32)
        ret = glr_sret

        #update acc
        if avg_eval:
            #ret = tf.cast(tf.greater_equal(tf.reduce_sum(ret,0, keepdims=True), np.ceil(len(lx)*0.5)), tf.int32)
            ret = tf.cast(tf.greater(tf.reduce_sum(ret,0, keepdims=True), 0.0), tf.int32)
            self.metrics['g12_acc'].update_state(y, ret)
        else:          
            self.metrics['g12_acc'].update_state(my, ret)
  
    @tf.function
    def evaluate_g1232(self, lx, ly, lry, x, y, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])
            my = tf.tile(y, [len(ly), 1])
            ay = tf.concat((ly, my), 1)
            ray = tf.concat((lry, my), 1)
        else:
            my = y
            ay = tf.concat((ly, y), 1)
            ray = tf.concat((lry, y), 1)

        batchsize = tf.shape(x)[0]
        labelednum = tf.shape(lx)[1]

        ax = tf.concat((lx,x),1)
        nodenum = tf.shape(ax)[1]
        unlabelednum = nodenum - labelednum

        dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
        edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

        #wnet1
        latents, shallows = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), False)
        latents = tf.reshape(latents, [batchsize, nodenum, -1])
        dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
        labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

        sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
        w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
        adj = tf.multiply(w, edge)

        glr_ret, glr_gs, glr_sgs = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

        denoise_ay = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
        denoise_ly = tf.slice(denoise_ay, [0,0], [batchsize, labelednum])

        #hnet
        glr_sign_ret = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
        ency = tf.one_hot(glr_sign_ret, 2) *  tf.math.abs(tf.expand_dims(glr_ret, 2))
        mau = tf.abs(tf.slice(glr_gs, [0,0], [batchsize, labelednum]) -  tf.slice(glr_sgs, [0,0], [batchsize, labelednum]))
        mau_mask = tf.cast(tf.greater(mau, kwargs.get('dynglr_thres', 2.0)), tf.float32)
        mau = tf.cast(tf.multiply(1.0 + tf.multiply(tf.reduce_max(mau, 1, keepdims=True) - mau, (kwargs.get('dynglr_tau', 3)-1)), mau_mask), tf.int32)
        mau = tf.concat((mau, tf.zeros((batchsize, unlabelednum), tf.int32)), 1)
        mau = tf.ones_like(mau, tf.int32)*kwargs.get('dynglr_tau', 3)

        dency = dynglr_knnfeature(ency, w, kwargs.get('dynglr_tau', 3), mau)
        dency = tf.reshape(dency, [batchsize * nodenum, -1])

        latents1, shallows1 = self.network_u([shallows, dency], False)
        latents1 = tf.reshape(latents1, [batchsize, nodenum, -1])
        dists1 = euclidean_distance(latents1, kwargs.get('use_fp16', False))
        labeled_dists1 = tf.slice(dists1, [0, 0,0], [batchsize, labelednum, labelednum])

        sparse_gamma = tf.cast(tf.greater(adj, 0.1), tf.float32)
        sparse_gamma = tf.cast(tf.reduce_sum(sparse_gamma, 2), tf.int32)
        edge1 = sparse_knn(dists1, sparse_gamma, nodenum)

        adj2 = edge1

        glr_ret2, _, _ = glr_fidelity(adj2, glr_ret, kwargs.get('dynglr_mu2', 1.0), labelednum, unlabelednum, nodenum, ally=True, normlap=kwargs.get('dynglr_norm', False), kappafactor=kwargs.get('kappa_factor', 0.5))

        glr_sret2 = tf.slice(glr_ret2, [0,labelednum], [batchsize, unlabelednum])
        #ret2 = tf.cast(tf.greater(glr_sret2, 0.0), tf.int32)
        ret2 = glr_sret2

        #update acc
        if avg_eval:
            #ret2 = tf.cast(tf.greater_equal(tf.reduce_sum(ret2,0, keepdims=True), np.ceil(len(lx)*0.5)), tf.int32)
            ret2 = tf.cast(tf.greater(tf.reduce_sum(ret2,0, keepdims=True), 0.0), tf.int32)
            self.metrics['g1232_acc'].update_state(y, ret2)
        else: 
            self.metrics['g1232_acc'].update_state(my, ret2)
    
    @tf.function
    def evaluate_g12312(self, lx, ly, lry, x, y, **kwargs):
        inputdim = np.shape(lx)[-1]
        avg_eval = len(x)==1 and len(lx)!=-1
        if avg_eval:
            x = tf.tile(x, [len(lx), 1, 1])
            my = tf.tile(y, [len(ly), 1])
            ay = tf.concat((ly, my), 1)
            ray = tf.concat((lry, my), 1)
        else:
            my = y
            ay = tf.concat((ly, y), 1)
            ray = tf.concat((lry, y), 1)

        batchsize = tf.shape(x)[0]
        labelednum = tf.shape(lx)[1]

        ax = tf.concat((lx,x),1)
        nodenum = tf.shape(ax)[1]
        unlabelednum = nodenum - labelednum

        dx = euclidean_distance(ax, kwargs.get('use_fp16', False), use_sqrt=True)
        edge = knn_edge(dx, nodenum, kwargs.get('dynglr_degree', 7), 1e-5, use_fp16=kwargs.get('use_fp16', False))

        #wnet1
        latents, shallows = self.network_w1(tf.reshape(ax, [batchsize*nodenum, inputdim]), False)
        latents = tf.reshape(latents, [batchsize, nodenum, -1])
        dists = euclidean_distance(latents, kwargs.get('use_fp16', False))
        labeled_dists = tf.slice(dists, [0, 0,0], [batchsize, labelednum, labelednum])

        sigma = adaptive_sigma(labeled_dists, ly, use_fp16=kwargs.get('use_fp16', False))
        w,_ = edge_weighting(dists, sigma, use_fp16=kwargs.get('use_fp16', False))
        adj = tf.multiply(w, edge)

        glr_ret, glr_gs, glr_sgs = glr_fidelity(adj, binarize(ly, 1), kwargs.get('dynglr_mu', 1.0), labelednum, unlabelednum, nodenum, normlap=kwargs.get('dynglr_norm', False))

        denoise_ay = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
        denoise_ly = tf.slice(denoise_ay, [0,0], [batchsize, labelednum])

        #hnet
        glr_sign_ret = tf.cast(tf.greater(glr_ret, 0.0), tf.int32)
        ency = tf.one_hot(glr_sign_ret, 2) *  tf.math.abs(tf.expand_dims(glr_ret, 2))
        mau = tf.abs(tf.slice(glr_gs, [0,0], [batchsize, labelednum]) -  tf.slice(glr_sgs, [0,0], [batchsize, labelednum]))
        mau_mask = tf.cast(tf.greater(mau, kwargs.get('dynglr_thres', 2.0)), tf.float32)
        mau = tf.cast(tf.multiply(1.0 + tf.multiply(tf.reduce_max(mau, 1, keepdims=True) - mau, (kwargs.get('dynglr_tau', 3)-1)), mau_mask), tf.int32)
        mau = tf.concat((mau, tf.zeros((batchsize, unlabelednum), tf.int32)), 1)
        mau = tf.ones_like(mau, tf.int32)*kwargs.get('dynglr_tau', 3)

        dency = dynglr_knnfeature(ency, w, kwargs.get('dynglr_tau', 3), mau)
        dency = tf.reshape(dency, [batchsize * nodenum, -1])

        latents1, shallows1 = self.network_u([shallows, dency], False)
        latents1 = tf.reshape(latents1, [batchsize, nodenum, -1])
        dists1 = euclidean_distance(latents1, kwargs.get('use_fp16', False))
        labeled_dists1 = tf.slice(dists1, [0, 0,0], [batchsize, labelednum, labelednum])

        sparse_gamma = tf.cast(tf.greater(adj, 0.1), tf.float32)
        sparse_gamma = tf.cast(tf.reduce_sum(sparse_gamma, 2), tf.int32)
        edge1 = sparse_knn(dists1, sparse_gamma, nodenum)

        #wnet2
        latents2 = self.network_w2([shallows, shallows1], False)
        latents2 = tf.reshape(latents2, [batchsize, nodenum, -1])
        dists2 = euclidean_distance(latents2, kwargs.get('use_fp16', False))
        labeled_dists2 = tf.slice(dists2, [0, 0,0], [batchsize, labelednum, labelednum])

        sigma2 = adaptive_sigma(labeled_dists2, denoise_ly, use_fp16=kwargs.get('use_fp16', False))
        w2,_ = edge_weighting(dists2, sigma2, use_fp16=kwargs.get('use_fp16', False))
        adj2 = tf.multiply(w2, edge1)

        glr_ret2, _, _ = glr_fidelity(adj2, glr_ret, kwargs.get('dynglr_mu2', 1.0), labelednum, unlabelednum, nodenum, ally=True, normlap=kwargs.get('dynglr_norm', False), kappafactor=kwargs.get('kappa_factor', 0.5))

        glr_sret2 = tf.slice(glr_ret2, [0,labelednum], [batchsize, unlabelednum])
        #ret2 = tf.cast(tf.greater(glr_sret2, 0.0), tf.int32)
        ret2 = glr_sret2

        #update acc
        if avg_eval:
            #ret2 = tf.cast(tf.greater_equal(tf.reduce_sum(ret2,0, keepdims=True), np.ceil(len(lx)*0.5)), tf.int32)
            ret2 = tf.cast(tf.greater(tf.reduce_sum(ret2,0, keepdims=True), 0.0), tf.int32)
            self.metrics['g12312_acc'].update_state(y, ret2)
        else:
            self.metrics['g12312_acc'].update_state(my, ret2)

    def destroy(self):
       tf.keras.backend.clear_session()
       del self.network_w1
       del self.network_u
       del self.network_w2
       del self.optimizer
       del self.optimizer1
       del self.optimizer2

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass
        self.network_w1.save_weights(path+'_w1.h5')
        self.network_u.save_weights(path+'_u.h5')
        self.network_w2.save_weights(path+'_w2.h5')

    def load(self, path):
        self.network_w1.load_weights(path+'_w1.h5')
        self.network_u.load_weights(path+'_u.h5')
        self.network_w2.load_weights(path+'_w2.h5')

    def load_g123(self, path):
        self.network_w1.load_weights(path+'_w1.h5')
        self.network_u.load_weights(path+'_u.h5')


def train_DYNGLR(path, trainx, trainy, trainry, validx, validy, validry, testx, testy, classes=10, **kwargs):
    batchsize = kwargs.get('batchsize', 16)
    graphnum = kwargs.get('graphnum', 1)
    graphsize = kwargs.get('graphsize', 120)
    samplesize = kwargs.get('samplesize', 30)
    batches = kwargs.get('batches', [320, 160, 120])
    validbatches = kwargs.get('validbatches', 10)
    fp16 = kwargs.get('use_fp16', False)

    model = DYNGLR(classes, **kwargs)

    train_set = GPUStratifiedRandomPack(trainx, trainy, trainry, classes, batchsize, samplesize, use_fp16=fp16)
    trainvalid_set = GPUStratifiedRandomPack(validx, validy, validry, classes, batchsize, int((graphsize-samplesize*classes)/2), use_fp16=fp16)

    traintest_set = GPUStratifiedRandomPack(trainx, trainy, trainry, classes, graphnum, samplesize, use_fp16=fp16)
    valid_set = GPURandomPack(validx, validy, validry, batchsize, graphsize-samplesize*classes, cycle=False, use_fp16=fp16)
    test_set = GPURandomPack(testx, testy, testy, True, graphsize-samplesize*classes, cycle=False, use_fp16=fp16)

    wnet1_train_loss_list = []
    wnet1_train_refloss_list = []
    
    hnet_train_loss_list = []
    hnet_train_refloss_list = []

    wnet2_train_loss_list = []

    wnet1_valid_acc_list =[]
    wnet1_valid_refacc_list = []

    hnet_valid_loss_list = []
    hnet_valid_refloss_list = []

    wnet2_valid_acc_list =[]
    wnet2_valid_refacc_list = []

    g2_acc = []
    g12_acc_list = []
    g1232_acc_list = []
    g12312_acc_list = []

    early_stopped = False

    train_set_iter = iter(train_set)
    trainvalid_set_iter = iter(trainvalid_set)

    #test epoch for g2
    traintest_set_iter = iter(traintest_set)
    for batch_x, batch_y, batch_ry in test_set:
        batch_lx, batch_ly, batch_lry = next(traintest_set_iter)
        model.evaluate_g2(batch_lx, batch_ly, batch_lry, batch_x, batch_y, **kwargs)

    g2_acc = model.metrics['g2_acc'].result().numpy()
    print('g2_acc=%f' % g2_acc)

    progbar = tqdm(range(1, batches[0]+1), ncols=120)
    for ibatch in progbar:
        #train step
        batch_lx, batch_ly, batch_lry = next(train_set_iter)
        batch_vx, batch_vy, batch_vy = next(trainvalid_set_iter)

        model.train_wnet1(batch_lx, batch_ly, batch_lry, batch_vx,  **kwargs)

        if (ibatch % validbatches) ==0:
            #query wnet1 train metrics
            wnet1_train_loss_list.append(model.metrics['wnet1_train_loss'].result().numpy())
            wnet1_train_refloss_list.append(model.metrics['wnet1_train_refloss'].result().numpy())
            model.metrics['wnet1_train_loss'].reset_states()
            model.metrics['wnet1_train_refloss'].reset_states()

            #validate epoch
            traintest_set.reset()
            traintest_set_iter = iter(traintest_set)
            for batch_x, batch_y, batch_ry in valid_set:
                batch_lx, batch_ly, batch_lry = next(traintest_set_iter)
                model.validate_wnet1(batch_lx, batch_ly, batch_lry, batch_x, batch_y, batch_ry, **kwargs)

            #query wnet1 valid metrics
            wnet1_valid_acc_list.append(model.metrics['wnet1_valid_acc'].result().numpy())
            wnet1_valid_refacc_list.append(model.metrics['wnet1_valid_refacc'].result().numpy())
            model.metrics['wnet1_valid_acc'].reset_states()
            model.metrics['wnet1_valid_refacc'].reset_states()

            #test epoch
            traintest_set.reset()
            traintest_set_iter = iter(traintest_set)
            for batch_x, batch_y, batch_ry in test_set:
                batch_lx, batch_ly, batch_lry = next(traintest_set_iter)
                model.evaluate_g12(batch_lx, batch_ly, batch_lry, batch_x, batch_y, **kwargs)

            #query wnet1 test metrics
            g12_acc_list.append(model.metrics['g12_acc'].result().numpy())
            model.metrics['g12_acc'].reset_states()

            progbar.set_description("wnet1_train_loss=%f, wnet1_valid_refacc=%f, g12_acc=%f" %(wnet1_train_loss_list[-1], wnet1_valid_refacc_list[-1], g12_acc_list[-1]))

    progbar1 = tqdm(range(1, batches[1]+1), ncols=120)
    for ibatch in progbar1:
        batch_lx, batch_ly, batch_lry = next(train_set_iter)
        batch_vx, batch_vy, batch_vy = next(trainvalid_set_iter)

        model.train_hnet(batch_lx, batch_ly, batch_lry, batch_vx,  **kwargs)

        if (ibatch % validbatches) ==0:
            hnet_train_loss_list.append(model.metrics['hnet_train_loss'].result().numpy())
            hnet_train_refloss_list.append(model.metrics['hnet_train_refloss'].result().numpy())
            model.metrics['hnet_train_loss'].reset_states()
            model.metrics['hnet_train_refloss'].reset_states()

            traintest_set.reset()
            traintest_set_iter = iter(traintest_set)
            for batch_x, batch_y, batch_ry in valid_set:
                batch_lx, batch_ly, batch_lry = next(traintest_set_iter)
                model.validate_hnet(batch_lx, batch_ly, batch_lry, batch_x, batch_y, batch_ry, **kwargs)

            hnet_valid_loss_list.append(model.metrics['hnet_valid_loss'].result().numpy())
            hnet_valid_refloss_list.append(model.metrics['hnet_valid_refloss'].result().numpy())

            model.metrics['hnet_valid_loss'].reset_states()
            model.metrics['hnet_valid_refloss'].reset_states()

            traintest_set.reset()
            traintest_set_iter = iter(traintest_set)
            for batch_x, batch_y, batch_ry in test_set:
                batch_lx, batch_ly, batch_lry = next(traintest_set_iter)
                model.evaluate_g1232(batch_lx, batch_ly, batch_lry, batch_x, batch_y, **kwargs)

            g1232_acc_list.append(model.metrics['g1232_acc'].result().numpy())
            model.metrics['g1232_acc'].reset_states()

            progbar1.set_description("hnet_train_loss=%f, hnet_valid_refloss=%f, g1232_acc=%f" %(hnet_train_loss_list[-1], hnet_valid_refloss_list[-1], g1232_acc_list[-1]))

    progbar2 = tqdm(range(1, batches[2]+1), ncols=120)
    for ibatch in progbar2:
        batch_lx, batch_ly, batch_lry = next(train_set_iter)
        batch_vx, batch_vy, batch_vy = next(trainvalid_set_iter)

        model.train_wnet2(batch_lx, batch_ly, batch_lry, batch_vx,  **kwargs)

        if (ibatch % validbatches) ==0:
            wnet2_train_loss_list.append(model.metrics['wnet2_train_loss'].result().numpy())
            model.metrics['wnet2_train_loss'].reset_states()

            traintest_set.reset()
            traintest_set_iter = iter(traintest_set)
            for batch_x, batch_y, batch_ry in valid_set:
                batch_lx, batch_ly, batch_lry = next(traintest_set_iter)
                model.validate_wnet2(batch_lx, batch_ly, batch_lry, batch_x, batch_y, batch_ry, **kwargs)

            wnet2_valid_acc_list.append(model.metrics['wnet2_valid_acc'].result().numpy())
            wnet2_valid_refacc_list.append(model.metrics['wnet2_valid_refacc'].result().numpy())
            model.metrics['wnet2_valid_acc'].reset_states()
            model.metrics['wnet2_valid_refacc'].reset_states()

            traintest_set.reset()
            traintest_set_iter = iter(traintest_set)
            for batch_x, batch_y, batch_ry in test_set:
                batch_lx, batch_ly, batch_lry = next(traintest_set_iter)
                model.evaluate_g12312(batch_lx, batch_ly, batch_lry, batch_x, batch_y, **kwargs)

            g12312_acc_list.append(model.metrics['g12312_acc'].result().numpy())
            model.metrics['g12312_acc'].reset_states()

            progbar2.set_description("wnet2_train_loss=%f, wnet2_valid_refacc=%f, g12312_acc=%f" %(wnet2_train_loss_list[-1], wnet2_valid_refacc_list[-1], g12312_acc_list[-1]))

    model.save(path)

    truth = []
    pred = []
    traintest_set.reset()
    traintest_set_iter = iter(traintest_set)
    for batch_x, batch_y, batch_ry in test_set:
        batch_lx, batch_ly, batch_lry = next(traintest_set_iter)
        pred.extend(model.predict(batch_lx, batch_ly, batch_x, **kwargs))
        truth.extend(batch_y)

    del train_set
    del trainvalid_set
    del traintest_set
    del valid_set
    del test_set

    model.destroy()
    del model

    return pred, truth, wnet1_train_loss_list, wnet1_train_refloss_list, hnet_train_loss_list, hnet_train_refloss_list,\
    wnet2_train_loss_list, wnet1_valid_acc_list, wnet1_valid_refacc_list, hnet_valid_loss_list, hnet_valid_refloss_list,\
    wnet2_valid_acc_list, wnet2_valid_refacc_list, g2_acc, g12_acc_list, g1232_acc_list, g12312_acc_list

def run_DYNGLR(log, trainx, trainy, trainry, validx, validy, validry, testx, testy, classes=2, **kwargs):
    if trainx.dtype==np.float16: trainx=trainx.astype(np.float32)
    if validx.dtype==np.float16: validx=validx.astype(np.float32)
    if testx.dtype==np.float16: testx=testx.astype(np.float32)

    pred, truth, wnet1_train_loss_list, wnet1_train_refloss_list, hnet_train_loss_list, hnet_train_refloss_list, wnet2_train_loss_list, \
        wnet1_valid_acc_list, wnet1_valid_refacc_list, hnet_valid_loss_list, hnet_valid_refloss_list, wnet2_valid_acc_list, wnet2_valid_refacc_list, \
        g2_acc, g12_acc_list, g1232_acc_list, g12312_acc_list =train_DYNGLR(log, trainx, trainy, trainry, validx, validy, validry, testx, testy, classes, **kwargs)

    return np.array(pred), np.array(truth), {\
        'wnet1_train_loss' : wnet1_train_loss_list, 'wnet1_train_refloss' : wnet1_train_refloss_list, 'hnet_train_loss' : hnet_train_loss_list, 'hnet_train_refloss' : hnet_train_refloss_list,\
    'wnet2_train_loss' : wnet2_train_loss_list, 'wnet1_valid_acc' : wnet1_valid_acc_list, 'wnet1_valid_refacc' : wnet1_valid_refacc_list, 'hnet_valid_loss' : hnet_valid_loss_list, 'hnet_valid_refloss' : hnet_valid_refloss_list,\
    'wnet2_valid_acc' : wnet2_valid_acc_list, 'wnet2_valid_refacc' : wnet2_valid_refacc_list, 'g2_acc' : g2_acc, 'g12_acc' : g12_acc_list, 'g1232_acc' : g1232_acc_list, 'g12312_acc' : g12312_acc_list}

