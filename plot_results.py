import pickle
import numpy as np
import matplotlib.pyplot as plt

noises = [0,5,10,15,20,25]
recs = {'g2': np.empty((len(noises),)), 'g12': np.empty((len(noises),)), 'g1232': np.empty((len(noises),)), 'g12312': np.empty((len(noises),))}

for ni, noise in enumerate(noises):
    all_g2_acc = []
    all_g12_acc = []
    all_g1232_acc = []
    all_g12312_acc = []
    for exp in range(20):
        with open('result/cifar10_transfer_binary/noise_'+str(noise) +'/dynglr/exp_'+str(exp)+'.meta', 'rb') as file:
            data = pickle.load(file)
            all_g2_acc.append(data['g2_acc'])
            all_g12_acc.append(data['g12_acc'][-1])
            all_g1232_acc.append(data['g1232_acc'][-1])
            all_g12312_acc.append(data['g12312_acc'][-1])


    recs['g2'][ni]=100.0-np.mean(all_g2_acc)*100.0
    recs['g12'][ni]=100.0-np.mean(all_g12_acc)*100.0
    recs['g1232'][ni]=100.0-np.mean(all_g1232_acc)*100.0
    recs['g12312'][ni]=100.0-np.mean(all_g12312_acc)*100.0

print(np.array2string(recs['g2'], precision=2, separator=' & '))
print(np.array2string(recs['g12'], precision=2, separator=' & '))
print(np.array2string(recs['g1232'], precision=2, separator=' & '))
print(np.array2string(recs['g12312'], precision=2, separator=' & '))


