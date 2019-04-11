### dense model

from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, Multiply,Activation, Reshape, Concatenate, Lambda

import numpy as np
from random import choice
from sklearn.model_selection import StratifiedKFold
from metrics import keras_auc,keras_precision,keras_recall,keras_f1,keras_fb
from common import read_data
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from collections import defaultdict

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
        super(LinkPredict, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        
        self.e1 = Embedding(input_dim[0],embedding_dim)
        self.e2 = Embedding(input_dim[1],embedding_dim)
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
        
        
        self.ls = [Concatenate(),
                   Dense(32,activation='relu'), 
                   Dense(1,activation='sigmoid'),
                   Reshape((-1,))]
        
        
        self.rate = 0.2
        if self.use_dp:
            self.dp = Dropout(self.rate)
            self.ls.insert(1, Dropout(self.rate))
            self.ls.insert(3, Dropout(self.rate))
            
        if self.use_bn:
            self.bn = BatchNormalization(axis=-1)

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
            
        if self.use_bn:
            s1 = self.bn(s1)
            p1 = self.bn(p1)
            o1 = self.bn(o1)
            
            s2 = self.bn(s2)
            p2 = self.bn(p2)
            o2 = self.bn(o2)
            
        l1 = [s1,p1,o1]
        l2 = [s2,p2,o2]
        x = [s3,o3]
        
        for layer in self.embedding:
            l1 = layer(l1)
            l2 = layer(l2)
        
        for layer in self.ls:
            x = layer(x)
        
        return l1,l2,x
    

def main(cv=False):

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
            distRc[mapping_cr[p]] += 1
        except KeyError:
            pass

    distRt = defaultdict(int)

    for s,p,o,score in Tg:
        X2.append((mapping_t[s],mapping_tr[p],mapping_t[o]))
        Y2.append(float(score))
        distRt[mapping_tr[p]] += 1

    CX1 = set(X1)
    TX2 = set(X2)

    Pc = [distRc[k] for k in sorted(distRc)]
    Pc = [i/sum(Pc) for i in Pc]

    Pt = [distRt[k] for k in sorted(distRt)]
    Pt = [i/sum(Pt) for i in Pt]


    while len(X1) < 5*len(Cg):
        s,p,o = mapping_c[choice(C)],np.random.choice(len(Rc),p=Pc),mapping_c[choice(C)]
        if (s,p,o) in CX1: continue
        X1.append((s,p,o))
        Y1.append(0)
        
    while len(X2) < 5*len(Tg):
        s,p,o = mapping_t[choice(T)],np.random.choice(len(Rt),p=Pt),mapping_t[choice(T)]
        if (s,p,o) in TX2: continue
        X2.append((s,p,o))
        Y2.append(0)

    X3tr = []
    Y3tr = []
    X3te = []
    Y3te = []

    for c,p,t,e in Etr:
        try:
            tmp = (mapping_c[c],mapping_er[p],mapping_t[t])
            e = float(e)
            X3tr.append(tmp)
            Y3tr.append(e)
        except KeyError:
            pass
    for c,p,t,e in Ete:
        try:
            tmp = (mapping_c[c],mapping_er[p],mapping_t[t])
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

    # Equal length inputs
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
    lossWeights = {"output_1": 1.0, "output_2": 1.0, "output_3": 1.0}
    num_epochs = 1000
    metrics = ['accuracy', keras_precision, keras_recall, keras_auc, keras_f1, keras_fb]
    callbacks = [EarlyStopping(monitor='output_3_loss', mode='min', patience=5, restore_best_weights=True)]
    results = []
    for mode in ['DistMult','TransE','HolE']:
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cvscores = []
        if cv:
            for train,test in kfold.split(X3tr,Y3tr):
                model = LinkPredict(input_dim=[N,M,Nr,Mr], embedding_dim=128, use_bn=True,use_dp=True, embedding_method=mode)
                # Compile model
                
                model.compile(loss=losses, loss_weights=lossWeights, optimizer='adagrad', metrics=metrics, callbacks=callbacks)
                # Fit the model
                
                tmp_xtr = X3tr[train]
                tmp_ytr = Y3tr[train]
                #oversample train input 3
                while len(tmp_ytr) < len(X1):
                    i = np.random.choice(len(tmp_ytr))
                    tmp_xtr = np.concatenate([tmp_xtr,tmp_xtr[i].reshape((1,3))], axis = 0)
                    tmp_ytr = np.append(tmp_ytr,tmp_ytr[i])
                
                model.fit([X1,X2,tmp_xtr], [Y1,Y2,tmp_ytr], epochs=num_epochs, batch_size=len(tmp_ytr), verbose=0)
                # evaluate the model
                
                scores = model.evaluate([X1[test],X2[test],X3tr[test]], [Y1[test],Y2[test],Y3tr[test]], verbose=0, batch_size=len(Y3tr[test]))
                
                cvscores.append(scores)
            cvscores = np.asarray(cvscores)
            for i,n in enumerate(model.metrics_names):
                print(mode,n,"%.2f (+/- %.2f)" % (np.mean(cvscores[:,i]), np.std(cvscores[:,i])))

        model = LinkPredict(input_dim=[N,M,Nr,Mr], embedding_dim=128, use_bn=True,use_dp=True, embedding_method=mode)
        model.compile('adagrad', loss=losses, loss_weights=lossWeights, metrics=metrics)
        model.fit([X1,X2,X3tr],[Y1,Y2,Y3tr], epochs = num_epochs, shuffle=True, verbose=2, callbacks=callbacks, batch_size = len(Y3tr))

        r1 = model.evaluate([X1te,X2te,X3te], [Y1te,Y2te,Y3te], batch_size=len(Y3te),verbose=0)
        for n,res in zip(model.metrics_names,r1):
            print(mode,n,res)
        
        r2 = model.predict([X1te,X2te,X3te])[-1]
        
        results.append([r1,r2,Y3te])
    
    return results
        


if __name__ == '__main__':
    main(cv=False)



















