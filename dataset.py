import numpy as np
import pickle as pkl

def getdata_byexp(dataset, noiselvl, exp):
    classes = 10
    if dataset=='cifar10':
        inpath = 'data/cifar10/cifar10_exp'+ str(exp) +'_noise'+ str(noiselvl if noiselvl else 0) +  '.pkl'    
    elif dataset=='cifar10_transfer':
        inpath = 'data/cifar10_transfer/cifar10_transfer_exp'+ str(exp) +'_noise'+ str(noiselvl if noiselvl else 0) +  '.pkl'    
    elif dataset=='cifar10_transfer_binary':
        inpath = 'data/cifar10_transfer_binary/cifar10_transfer_binary_exp'+ str(exp) +'_noise'+ str(noiselvl if noiselvl else 0) +  '.pkl'
        classes = 2
    elif dataset=='mnist':
        inpath = 'data/mnist/mnist_exp'+ str(exp) +'_noise'+ str(noiselvl if noiselvl else 0) +  '.pkl'    
    elif dataset=='mnist_binary':
        inpath = 'data/mnist_binary/mnist_binary_exp'+ str(exp) +'_noise'+ str(noiselvl if noiselvl else 0) +  '.pkl'    
        classes = 2
    
    with open(inpath, 'rb') as ifs:
        xy = pkl.load(ifs)
        return np.array(xy['trainx']), np.array(xy['trainy']), np.array(xy['trainry']), np.array(xy['validx']), np.array(xy['validy']), np.array(xy['validry']), np.array(xy['testx']), np.array(xy['testy']), classes

def gettestdata(dataset):
    if dataset=='cifar10_transfer_binary':
        inpath = 'data/cifar10_transfer_binary/cifar10_transfer_binary.pkl'
        classes = 2
    elif dataset=='cifar10_1_transfer_binary':
        inpath = 'data/cifar10_1_transfer_binary/cifar10_1_transfer_binary.pkl'
        classes = 2

    with open(inpath, 'rb') as ifs:
        xy = pkl.load(ifs)
        return np.array(xy['testx']), np.array(xy['testy']), classes
