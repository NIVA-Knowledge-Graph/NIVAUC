### dense model

from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, Multiply,Activation, Reshape, Concatenate, Lambda

from keras.utils import plot_model

import numpy as np
from random import choice
from sklearn.model_selection import StratifiedKFold
from metrics import keras_auc,keras_precision,keras_recall,keras_f1,keras_fb
from common import read_data
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from collections import defaultdict
from keras.constraints import MaxNorm
from tqdm import tqdm


def circ(s,o):
    s,o = tf.cast(s, dtype=tf.complex128),tf.cast(o,dtype=tf.complex128)
    return tf.real(tf.spectral.ifft(tf.conj(tf.spectral.fft(s))*tf.spectral.fft(o)))

def circular_cross_correlation(x, y):
    """Periodic correlation, implemented using the FFT.
    x and y must be of the same length.
    """
    return tf.real(tf.ifft(tf.multiply(tf.conj(tf.fft(tf.cast(x, tf.complex64))) , tf.fft(tf.cast(y, tf.complex64)))))

def HolE(s,p,o):
    # sigm(p^T (s \star o))
    # dot product in tf: sum(multiply(a, b) axis = 1)
    return tf.reduce_sum(tf.multiply(p, circular_cross_correlation(s, o)), axis = 1)

def TransE(s,p,o):
    return 1/tf.norm(s+p-o, axis = 1)

class LinkPredict(Model):

    def __init__(self,input_dim, embedding_dim = 128, use_bn=False, use_dp=False, embedding_method = 'DistMult'):
        super(LinkPredict, self).__init__(name='lp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        
        if embedding_method == 'HolE':
            constraint = MaxNorm(1,axis=1)
        elif embedding_method == 'TransE':
            constraint = None
        elif embedding_method == 'DistMult':
            constraint = MaxNorm(1,axis=1)
        
        self.e1 = Embedding(input_dim[0],embedding_dim,embeddings_constraint=constraint)
        self.e2 = Embedding(input_dim[1],embedding_dim,embeddings_constraint=constraint)
        self.r1 = Embedding(input_dim[2],embedding_dim)
        self.r2 = Embedding(input_dim[3],embedding_dim)
        
        
        if embedding_method == 'DistMult':
            self.embedding = [Multiply(),
                            Lambda(lambda x: K.sum(x, axis=-1)),
                            Activation('sigmoid'),
                            Reshape((-1,))]
        elif embedding_method == 'HolE':
            self.embedding = [Lambda(lambda x: HolE(x[0],x[1],x[2])),
                            Activation('sigmoid'),
                            Reshape((-1,))]
        elif embedding_method == 'TransE':
            self.embedding = [Lambda(lambda x: TransE(x[0],x[1],x[2])),
                              Activation('tanh'),
                              Reshape((-1,))]
            
        else:
            raise NotImplementedError(embedding_method+' not implemented')
        
        
        self.rate = 0.2
        
        self.ls = [Concatenate(),
                   Dense(32),
                   Dropout(self.rate),
                   BatchNormalization(axis=-1),
                   Activation('relu'),
                   Dense(1,activation='sigmoid'),
                   Reshape((-1,))]
        

        if self.use_dp:
            self.dp = Dropout(self.rate)
            
        if self.use_bn:
            self.bn1 = BatchNormalization(axis=-1)
            self.bn2 = BatchNormalization(axis=-1)
            

    def call(self, inputs):
        # triple1, triple2, triple3
        t1,t2,t3 = inputs
        
        s1,p1,o1 = t1[:,0],t1[:,1],t1[:,2]
        s2,p2,o2 = t2[:,0],t2[:,1],t2[:,2]
        s3,p3,o3 = t3[:,0],t3[:,1],t3[:,2]
        
        s1 = self.e1(s1)
        p1 = self.r1(p1)
        o1 = self.e1(o1)
        
        s2 = self.e2(s2)
        p2 = self.r2(p2)
        o2 = self.e2(o2)
        
        s3 = self.e1(s3)
        o3 = self.e2(o3)
        
        if self.use_dp:
            s1 = self.dp(s1)
            p1 = self.dp(p1)
            o1 = self.dp(o1)
            
            s2 = self.dp(s2)
            p2 = self.dp(p2)
            o2 = self.dp(o2)
            
            s3 = self.dp(s3)
            o3 = self.dp(o3)
            
        if self.use_bn:
            s1 = self.bn1(s1)
            p1 = self.bn1(p1)
            o1 = self.bn1(o1)
            
            s2 = self.bn2(s2)
            p2 = self.bn2(p2)
            o2 = self.bn2(o2)
            
            s3 = self.bn1(s3)
            o3 = self.bn2(o3)
            
        l1 = [s1,p1,o1]
        l2 = [s2,p2,o2]
        x = [s3,o3]
        
        for layer in self.embedding:
            l1 = layer(l1)
            l2 = layer(l2)
        
        for layer in self.ls:
            x = layer(x)
        
        return l1,l2,x
    

def main(cv=False, verbose=2, file_number = 0, repeat = 0):

    Cg = read_data('./data/chemical_graph.txt')
    Cs = read_data('./data/chemical_similarity.txt')
    Tg = read_data('./data/taxonomy_graph.txt')
    Etr = read_data('./data/endpoints_train.txt')
    Ete = read_data('./data/endpoints_test.txt')

    C = set.union(*[set([a,b]) for a,_,b,_ in Cg])
    T = set.union(*[set([a,b]) for a,_,b,_ in Tg])
    Rc = set([r for _,r,_,_ in Cg]) | set([r for _,r,_,_ in Cs])
    Rt = set([r for _,r,_,_ in Tg])
    Re = set([r for _,r,_,_ in Etr]) | set([r for _,r,_,_ in Ete]) 

    N, M, Nr, Mr = len(C),len(T),len(Rc),len(Rt)

    mapping_c = {c:i for c,i in zip(C,range(len(C)))}
    mapping_t = {c:i for c,i in zip(T,range(len(T)))}
    mapping_cr = {c:i for c,i in zip(Rc,range(len(Rc)))}
    mapping_tr = {c:i for c,i in zip(Rt,range(len(Rt)))}
    mapping_er = {c:i for c,i in zip(Re,range(len(Re)))}

    X1 = []
    X2 = []
    X3 = []
    Y1 = []
    Y2 = []
    Y3 = []

    distRc = defaultdict(int)

    C,T,Rc,Rt = list(C),list(T),list(Rc),list(Rt)

    for s,p,o,score in Cg:
        X1.append((mapping_c[s],mapping_cr[p],mapping_c[o]))
        Y1.append(float(score))
        
        X1.append((mapping_c[choice(C)],mapping_cr[p],mapping_c[choice(C)]))
        Y1.append(0)
        
        distRc[mapping_cr[p]] += 1
        
    for s,p,o,score in Cs:
        try:
            score = float(score)
            if score > 0.5:
                score = 1
            else:
                continue
            X1.append((mapping_c[s],mapping_cr[p],mapping_c[o]))
            Y1.append(float(score))
            
            X1.append((mapping_c[choice(C)],mapping_cr[p],mapping_c[choice(C)]))
            Y1.append(0)
            
        except KeyError:
            pass

    distRt = defaultdict(int)

    for s,p,o,score in Tg:
        X2.append((mapping_t[s],mapping_tr[p],mapping_t[o]))
        Y2.append(float(score))
        
        X2.append((mapping_t[choice(T)],mapping_tr[p],mapping_t[choice(T)]))
        Y2.append(0)
        
    X3tr = []
    Y3tr = []
    X3te = []
    Y3te = []

    for c,p,t,e in Etr:
        try:
            tmp = (mapping_c[c],mapping_er[p],mapping_t[t])
            if tmp in X3tr: continue
            e = float(e)
            X3tr.append(tmp)
            Y3tr.append(e)
            
        except KeyError:
            pass
    for c,p,t,e in Ete:
        try:
            tmp = (mapping_c[c],mapping_er[p],mapping_t[t])
            if tmp in X3te: continue
            e = float(e)
            X3te.append(tmp)
            Y3te.append(e)
            
        except KeyError:
            pass

    #Oversampling training
    u,c = np.unique(Y3tr, return_counts=True)
    while c[0] != c[1]:
        idx = np.random.choice(len(Y3tr))
        if Y3tr[idx] == u[np.argmin(c)]:
            Y3tr.append(Y3tr[idx])
            X3tr.append(X3tr[idx])
        u,c = np.unique(Y3tr, return_counts=True)

    d = len(Y1)/len(Y3tr)

    # Equal length inputs
    if not cv:
        m = max(len(Y1),len(Y2),len(Y3tr))
        while min(len(Y1),len(Y2),len(Y3tr)) != m:
            if len(Y1) < m:
                idx = np.random.choice(len(Y1))
                Y1.append(Y1[idx])
                X1.append(X1[idx])
            if len(Y2) < m:
                idx = np.random.choice(len(Y2))
                Y2.append(Y2[idx])
                X2.append(X2[idx])
            if len(Y3tr) < m:
                idx = np.random.choice(len(Y3tr))
                Y3tr.append(Y3tr[idx])
                X3tr.append(X3tr[idx])

    X1,X2,X3tr = np.asarray(X1),np.asarray(X2),np.asarray(X3tr)
    Y1,Y2,Y3tr = np.asarray(Y1),np.asarray(Y2),np.asarray(Y3tr)

    #Undersampling test to make all inputs equal length.
    m = len(Y3te)
    X1te,X2te,X3te = X1[:m],X2[:m],X3te[:m]
    Y1te,Y2te,Y3te = Y1[:m],Y2[:m],Y3te[:m]
    X1te,X2te,X3te = np.asarray(X1te),np.asarray(X2te),np.asarray(X3te)
    Y1te,Y2te,Y3te = np.asarray(Y1te),np.asarray(Y2te),np.asarray(Y3te)

    losses = {"output_1": "binary_crossentropy",
            "output_2": "binary_crossentropy",
            "output_3": "binary_crossentropy"
                }
    lW = {}
    lW['DistMult'] = {"output_1": 0.5, "output_2": 0.5, "output_3": 1.0}
    lW['HolE'] = {"output_1": 0.5, "output_2": 0.5, "output_3": 1.0}
    lW['TransE'] = {"output_1": 1.0, "output_2": 1.0, "output_3": 1.0}
    
    num_epochs = 1000
    metrics = ['accuracy', keras_precision, keras_recall, keras_auc, keras_f1, keras_fb]
    callbacks = [EarlyStopping(monitor='output_3_loss', mode='min', patience=5, restore_best_weights=True)]
    results = {}

    for mode in ['TransE','DistMult','HolE']:
        lossWeights = lW[mode]
        
        if cv:
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            cvscores = []
            with tqdm(total=10,desc='CV: '+mode) as pbar:
                for train,test in kfold.split(X3tr,Y3tr):
                    model = LinkPredict(input_dim=[N,M,Nr,Mr], embedding_dim=128, use_bn=True,use_dp=True, embedding_method=mode)
                    # Compile model
                    
                    model.compile(loss=losses, loss_weights=lossWeights, optimizer='adagrad', metrics=metrics, callbacks=callbacks)
                    # Fit the model
                    
                    tmpx3 = X3tr[train]
                    tmpy3 = Y3tr[train]
                    
                    tmpx1 = X1
                    tmpy1 = Y1
                    tmpx2 = X2
                    tmpy2 = Y2
                    
                    m = max(len(tmpy1),len(tmpy2),len(tmpy3))
                    while min(len(tmpy1),len(tmpy2),len(tmpy3)) != m:
                        if len(tmpy1) < m:
                            idx = np.random.choice(len(tmpy1))
                            tmpx1 = np.concatenate((tmpx1,tmpx1[idx].reshape((1,3))),axis=0)
                            tmpy1 = np.append(tmpy1,tmpy1[idx])
                        if len(tmpy2) < m:
                            idx = np.random.choice(len(tmpy2))
                            tmpx2 = np.concatenate((tmpx2,tmpx2[idx].reshape((1,3))),axis=0)
                            tmpy2 = np.append(tmpy2,tmpy2[idx])
                        if len(tmpy3) < m:
                            idx = np.random.choice(len(tmpy3))
                            tmpx3 = np.concatenate((tmpx3,tmpx3[idx].reshape((1,3))),axis=0)
                            tmpy3 = np.append(tmpy3,tmpy3[idx])
                        
                    model.fit([tmpx1,tmpx2,tmpx3], [tmpy1,tmpy2,tmpy3], epochs=num_epochs, batch_size=len(tmpy3), verbose=verbose, validation_split=0.0, shuffle=True)
                    
                    
                    # evaluate the model
                    
                    scores = model.evaluate([X1[test],X2[test],X3tr[test]], [Y1[test],Y2[test],Y3tr[test]], verbose=0, batch_size=len(Y3tr[test]))
                    
                    cvscores.append(scores)
                    
                    pbar.update(1)
            cvscores = np.asarray(cvscores)
            for i,n in enumerate(model.metrics_names):
                print(mode,n,"%.2f (+/- %.2f)" % (np.mean(cvscores[:,i]), np.std(cvscores[:,i])))
        
        else:
            model = LinkPredict(input_dim=[N,M,Nr,Mr], embedding_dim=128, use_bn=True,use_dp=True, embedding_method=mode)
            model.compile('adagrad', loss=losses, loss_weights=lossWeights, metrics=metrics)
            
            if repeat:
                X1 = np.repeat(X1,repeat,axis=0)
                X2 = np.repeat(X2,repeat,axis=0)
                X3tr = np.repeat(X3tr,repeat,axis=0)
                Y1 = np.repeat(Y1,repeat,axis=0)
                Y2 = np.repeat(Y2,repeat,axis=0)
                Y3tr = np.repeat(Y3tr,repeat,axis=0)
                num_epochs = int(num_epochs/repeat)
            
            model.fit([X1,X2,X3tr],[Y1,Y2,Y3tr], epochs = num_epochs, shuffle=True, verbose=verbose, callbacks=callbacks, batch_size = len(Y3tr),validation_split=0.0)

            r1 = model.evaluate([X1te,X2te,X3te], [Y1te,Y2te,Y3te], batch_size=256,verbose=0)
            results[mode] = r1
            
            p = model.predict([X1te,X2te,X3te])[-1]
            with open('./results/LP/'+mode+'/'+str(file_number)+'.txt', 'w') as f:
                for a,b in zip(Y3te,p):
                    f.write(str(a)+','+str(b[-1])+'\n')
        
    return model, results

if __name__ == '__main__':
    main(cv=True,verbose=0)
    num = 10
    cvscores = defaultdict(list)
    for i in tqdm(range(num)):
        model, scores = main(cv=False, verbose=0, file_number=i, repeat = 0)
        for k in scores:
            cvscores[k].append(scores[k])
        
    for k in cvscores:
        scores = np.asarray(cvscores[k])
        for i,n in enumerate(model.metrics_names):
            print(k, n,"%.2f (+/- %.2f)" % (np.mean(scores[:,i]), np.std(scores[:,i])))



















