import pickle
from dataset import *
from models.tensorflow.dynglr import run_DYNGLR
from utils import *


def run(method, dataset, exp, noise, logmethod=None, **kwargs):
    trainx, trainy, trainry, validx, validy, validry, testx, testy, classes = getdata_byexp(dataset, noise, exp)

    if logmethod is None: logmethod= method

    log = 'model/' + dataset + '/noise_'+str(noise)+'/' + logmethod + '/exp_' + str(exp)
    out = 'result/' + dataset + '/noise_'+str(noise)+'/' + logmethod + '/exp_' + str(exp)

    if method=='dynglr':
        predy, truthy, meta = run_DYNGLR(log, trainx, trainy, trainry, validx, validy, validry, testx, testy, classes, **kwargs)

    logsuffix = kwargs.get('logsuffix', '')
    return savemetrics(out+logsuffix, predy.flat, truthy.flat, ["class "+str(i) for i in range(classes)], meta=meta)
