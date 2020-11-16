from __future__ import print_function
import numpy as np
import pickle as pkl

def train_vald_split(y, train, rng, classes=10):

    inds = [None]*classes
    sinds = [None]*classes
    num_train = [None]*classes

    for i in range(classes):
        inds[i] = np.where(y==i)[0]
        sinds[i] = inds[i][rng.permutation(len(inds[i]))]
        num_train[i] = int(len(inds[i])* train)
    
    itrain = np.concatenate([ind[:num_train[i]] for i,ind in enumerate(sinds)])
    ivalid = np.concatenate([ind[num_train[i]:] for i,ind in enumerate(sinds)])

    return itrain, ivalid

def confusey(y, noise, rng, classes=10):
    ret = np.copy(y)
    if noise == 0:
        return ret

    inds = [None]*classes
    sinds = [None]*classes
    sels = [None]*classes

    for i in range(classes):
        inds[i] = np.where(y==i)[0]
        sinds[i] = inds[i][rng.permutation(len(inds[i]))]

    for i in range(classes):
        num = max((int(len(sinds[i])* noise  / 100.0), classes-1))
        iinds = sinds[i][:num]

        chs = np.arange(classes)
        chs = np.delete(chs, i)

        ret[iinds] = rng.choice(chs, (num,))

    return ret

def load(path):
    with open(path, 'rb') as ofs:
        data= pkl.load(ofs)
        return data['trainx'], data['trainy'], data['testx'], data['testy']


def dump(path, trainx, trainy, trainry, validx, validy, validry, testx, testy, ishalf=True):
    if ishalf:
        trainx = trainx.astype('float16')
        validx = validx.astype('float16')
        testx = testx.astype('float16')

    with open(path, 'wb') as ofs:
        pkl.dump({'trainx':trainx, 'trainy':np.reshape(trainy, (-1,)), 'trainry':np.reshape(trainry, (-1,)), 'validx':validx, 'validy':np.reshape(validy, (-1,)), 'validry':np.reshape(validry, (-1,)), 'testx': testx, 'testy': np.reshape(testy, (-1,))}, ofs)    

datadir = 'data/'
dataset = 'cifar10_transfer_binary'

exps = 20
trainp = 0.9

c1 = 0  #airplance
c2 = 8  #ship

randstates = np.arange(exps, dtype=np.int)

x_train, y_train, x_test, y_test = load('data/cifar10_transfer.pkl')

inds1 = np.where(y_train==c1)[0]
inds2 = np.where(y_train==c2)[0]

x_train = np.concatenate((x_train[inds1], x_train[inds2]), 0)
y_train = np.concatenate((np.ones((len(inds1),), np.int32), np.zeros((len(inds1),), np.int32)), 0)

inds3 = np.where(y_test==c1)[0]
inds4 = np.where(y_test==c2)[0]

x_test = np.concatenate((x_test[inds3], x_test[inds4]), 0)
y_test = np.concatenate((np.ones((len(inds3),), np.int32), np.zeros((len(inds4),), np.int32)), 0)

print(dataset, ',', np.shape(y_train)[0], ',', np.shape(y_test)[0])

for noise in [0, 5, 10, 15, 20, 25]:
    for exp in range(exps):
        pklfile = datadir + dataset + '/' + dataset.lower() + '_exp' + str(exp) + '_noise' + str(noise) + '.pkl'

        rng = np.random.RandomState(exp)

        itrain, ivalid = train_vald_split(y_train, trainp, rng, 2)

        rng.shuffle(itrain)
        rng.shuffle(ivalid)

        trainx = x_train[itrain]
        validx = x_train[ivalid]

        trainry = y_train[itrain]
        validry = y_train[ivalid]

        trainny = confusey(trainry, noise, rng, 2)
        validny = confusey(validry, noise, rng, 2)

        print('train acc=', np.mean(np.equal(trainny, trainry).astype(np.float)))
        print('valid acc=', np.mean(np.equal(validny, validry).astype(np.float)))

        print(np.shape(trainx))
        print(np.shape(trainry))
        print(np.shape(trainny))
        print(np.shape(validx))
        print(np.shape(validry))
        print(np.shape(validny))
        print(np.shape(x_test))
        print(np.shape(y_test))

        dump(pklfile, trainx, trainny, trainry, validx, validny, validry, x_test, y_test)
        print(pklfile, ' saved')