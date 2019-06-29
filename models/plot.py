#plot.py

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,roc_auc_score,fbeta_score
import numpy as np
from collections import defaultdict

def load_data(directory,num=10,avg_first = True):
    Y = []
    Y_hat = []
    for i in range(num):
        tmp1 = []
        tmp2 = []
        with open(directory+str(i)+'.txt') as f:
            for l in f:
                l = l.strip()
                a,b = l.split(',')
                a,b = float(a),float(b)
                tmp1.append(a)
                tmp2.append(b)
        Y.append(tmp1)
        Y_hat.append(tmp2)
    

    Y = np.asarray(Y)
    Y_hat = np.asarray(Y_hat)

    Y = np.mean(Y,axis=0)
    if avg_first:
        Y_hat = np.mean(Y_hat,axis=0)
        Y_hat = np.expand_dims(Y_hat,axis=0)
    
    return Y,Y_hat
        

def my_round(x,th):
    a = np.around(x-th+0.5)
    a = [min(1,b) for b in a]
    return a 
    
Y,Y_hat = {},{}

num_files = 10
avg_first = False

di = './results/MLP/'
Y['M2'],Y_hat['M2'] = load_data(di, num_files, avg_first=avg_first)
di = './results/LP/TransE/'
Y['M2 (TransE)'],Y_hat['M2 (TransE)'] = load_data(di,num_files,avg_first=avg_first)
di = './results/LP/DistMult/'
Y['M2 (DistMult)'],Y_hat['M2 (DistMult)'] = load_data(di,num_files,avg_first=avg_first)
di = './results/LP/HolE/'
Y['M2 (HolE)'],Y_hat['M2 (HolE)'] = load_data(di,num_files,avg_first=avg_first)


th = 0.5

results = defaultdict(list)

for k in Y:
    a = []
    r = []
    p = []
    f1 = []
    f2 = []
    auc = []
    for s in Y_hat[k]:
        a.append(accuracy_score(Y[k],my_round(s,th)))
        r.append(recall_score(Y[k],my_round(s,th)))
        p.append(precision_score(Y[k],my_round(s,th)))
        f1.append(f1_score(Y[k],my_round(s,th)))
        f2.append(fbeta_score(Y[k],my_round(s,th),2))
        auc.append(roc_auc_score(Y[k],s))
        
    a_mean = np.mean(np.asarray(a),axis=0)
    r_mean = np.mean(np.asarray(r),axis=0)
    p_mean = np.mean(np.asarray(p),axis=0)
    f1_mean = np.mean(np.asarray(f1),axis=0)
    f2_mean = np.mean(np.asarray(f2),axis=0)
    auc_mean = np.mean(np.asarray(auc),axis=0)
    
    a_std = np.std(np.asarray(a),axis=0)
    r_std = np.std(np.asarray(r),axis=0)
    p_std = np.std(np.asarray(p),axis=0)
    f1_std = np.std(np.asarray(f1),axis=0)
    f2_std = np.std(np.asarray(f2),axis=0)
    auc_std = np.std(np.asarray(auc),axis=0)
    
    print(k,'Accuracy: ','{0} +- {1}'.format(round(a_mean,2),round(a_std,2)))
    print(k,'Precision: ','{0} +- {1}'.format(round(p_mean,2),round(p_std,2)))
    print(k,'Recall: ','{0} +- {1}'.format(round(r_mean,2),round(r_std,2)))
    print(k,'F1: ','{0} +- {1}'.format(round(f1_mean,2),round(f1_std,2)))
    print(k,'F2: ','{0} +- {1}'.format(round(f2_mean,2),round(f2_std,2)))
    print(k,'AUC: ','{0} +- {1}'.format(round(auc_mean,2),round(auc_std,2)))
    
    
acc_mean = defaultdict(list)
rec_mean = defaultdict(list)
pres_mean = defaultdict(list)
acc_std = defaultdict(list)
rec_std = defaultdict(list)
pres_std = defaultdict(list)

ths = np.arange(0,1+0.01,0.01)
for th in ths:
    for k in Y:
        a = []
        r = []
        p = []
        for s in Y_hat[k]:
            tmp = my_round(s,th)
            a.append(accuracy_score(Y[k],tmp))
            r.append(recall_score(Y[k],tmp))
            p.append(precision_score(Y[k],tmp))
            
        a_mean = np.mean(np.asarray(a),axis=0)
        r_mean = np.mean(np.asarray(r),axis=0)
        p_mean = np.mean(np.asarray(p),axis=0)
        a_std = np.std(np.asarray(a),axis=0)
        r_std = np.std(np.asarray(r),axis=0)
        p_std = np.std(np.asarray(p),axis=0)
        
        acc_mean[k].append(a_mean)
        rec_mean[k].append(r_mean)
        pres_mean[k].append(p_mean)
        acc_std[k].append(a_std)
        rec_std[k].append(r_std)
        pres_std[k].append(p_std)

ps = []
for k in acc_mean:
    a, = plt.plot(ths,acc_mean[k])
    ps.append(a)

plt.legend(ps,list(acc_mean.keys()),fontsize=20)
plt.tick_params(labelsize=20)
plt.xlabel('Threshold',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.show()

ps = []
for k in rec_mean:
    a, = plt.plot(ths,rec_mean[k])
    ps.append(a)

plt.legend(ps,list(rec_mean.keys()),fontsize=20)
plt.tick_params(labelsize=20)
plt.xlabel('Threshold',fontsize=20)
plt.ylabel('Recall',fontsize=20)
plt.show()

ps = []
for k in pres_mean:
    a, = plt.plot(ths,pres_mean[k])
    ps.append(a)

plt.legend(ps,list(pres_mean.keys()),fontsize=20)
plt.tick_params(labelsize=20)
plt.xlabel('Threshold',fontsize=20)
plt.ylabel('Precision',fontsize=20)
plt.show()

