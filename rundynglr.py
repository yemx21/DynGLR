from console import *
import argparse

noise_batchsize = 16
noises = [0, 5, 10, 15, 20, 25]
noise_mu = None
noise_mu2 = None

gpu_memory=2048

def run_dynglr(gpus):

    graphsize=120
    samplesize=30

    dynglr_degree = 35
    dynglr_margin = 10.0

    dynglr_thres = 1.0
    dynglr_thres2 = 0.6

    dynglr_tau =  3
    graphnum = 15

    kappa_factor = 1.0

    batches = [240, 160, 240]
    lr_mode_stages = ['piecewise_constant', 'piecewise_constant', 'piecewise_constant']
    lrs_stages= [[0.001, 0.0005, 0.0001], [0.001, 0.0005, 0.0001], [0.002, 0.001, 0.0005, 0.0001]]
    lr_stairs_stages = [[60, 120, 180], [60, 90, 120], [60, 90, 120, 180]]

    dynglr_norm =True
    g2_normlap = False

    for noise in noises:
        for exp in range(20):
            run('dynglr', 'cifar10_transfer_binary', exp, noise, 'dynglr', randstate=exp, graphsize=graphsize, samplesize=samplesize, batches=batches, graphnum=graphnum, kappa_factor=kappa_factor, dynglr_tau=dynglr_tau, dynglr_norm=dynglr_norm, lr_mode_stages=lr_mode_stages, lrs_stages=lrs_stages, lr_stairs_stages=lr_stairs_stages, batchsize=noise_batchsize, dynglr_degree=dynglr_degree, dynglr_margin=dynglr_margin,  dynglr_mu=noise_mu, dynglr_mu2=noise_mu2, dynglr_thres=dynglr_thres, dynglr_thres2=dynglr_thres2, g2_normlap=g2_normlap, gpus=gpus, gpu_memory=gpu_memory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'dynglr')
    parser.add_argument('--gpus', help='specify gpus', default=0, type=int)
    args = parser.parse_args()

    run_dynglr([args.gpus])
