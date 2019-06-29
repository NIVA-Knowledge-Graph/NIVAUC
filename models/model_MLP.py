### dense model

from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization, Reshape, Embedding, Reshape, Concatenate
import numpy as np
from sklearn.model_selection import StratifiedKFold
from metrics import keras_auc,keras_precision,keras_recall,keras_f1,keras_fb
from common import prep, read_data
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.optimizers import SGD
from tqdm import tqdm

from keras.constraints import MaxNorm

class SimpleMLP(Model):

    def __init__(self, input_dim, embedding_dim = [128,128], use_bn=False, use_dp=False):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        
        if not isinstance(embedding_dim,list):
            embedding_dim = [embedding_dim] * 2
            
        constraint = MaxNorm(1,axis=1)
        
        self.e1 = Embedding(input_dim[0],embedding_dim[0],embeddings_constraint=constraint)
        self.e2 = Embedding(input_dim[1],embedding_dim[1],embeddings_constraint=constraint)
        self.m = Concatenate()
        
        self.l1 = Dense(32, activation='relu')
        
        self.ac = Dense(1, activation='sigmoid')
        
        if self.use_dp:
            self.dp = Dropout(0.2)
        if self.use_bn:
            self.bn = BatchNormalization(axis=-1)

    def call(self, inputs):
        x1,x2 = inputs[:,0],inputs[:,1]
        
        x1 = self.e1(x1)
        x2 = self.e2(x2)
        
        x = self.m([x1,x2])
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            self.bn = BatchNormalization(axis=-1)
        
        x = self.l1(x)
        
        if self.use_dp:
            x = self.dp(x)
            
        x = self.ac(x)
        return x
    
def main(cv=False, verbose = 1, num_cl=10):
    Etr = read_data('./data/endpoints_train.txt')
    Ete = read_data('./data/endpoints_test.txt')

    Cg = read_data('./data/chemical_graph.txt')
    Tg = read_data('./data/taxonomy_graph.txt')

    C = set.union(*[set([a,b]) for a,_,b,_ in Cg])
    T = set.union(*[set([a,b]) for a,_,b,_ in Tg])

    Xc = set([k for k,_,_,_ in Etr if k in C]) | set([k for k,_,_,_ in Ete if k in C])
    Xt = set([k for _,_,k,_ in Etr if k in T]) | set([k for _,_,k,_ in Ete if k in T])
    N = len(Xc)
    M = len(Xt)
    mapping_c = {c:i for c,i in zip(Xc,range(N))}
    mapping_t = {c:i for c,i in zip(Xt,range(M))}

    Xtr = []
    Ytr = []
    Xte = []
    Yte = []

    for c,p,t,e in Etr:
        try:
            tmp = [mapping_c[c],mapping_t[t]]
            if tmp in Xtr: continue
            e = float(e)
            Xtr.append(tmp)
            Ytr.append(e)
        except KeyError:
            pass
            
    for c,p,t,e in Ete:
        try:
            tmp = [mapping_c[c],mapping_t[t]]
            if tmp in Xte: continue
            e = float(e)
            Xte.append(tmp)
            Yte.append(e)
        except KeyError:
            pass

    #Oversampling training
    #u,c = np.unique(Ytr, return_counts=True)
    #while c[0] != c[1]:
        #idx = np.random.choice(len(Ytr))
        #if Ytr[idx] == u[np.argmin(c)]:
            #Ytr.append(Ytr[idx])
            #Xtr.append(Xtr[idx])
        #u,c = np.unique(Ytr, return_counts=True)

    Xtr = np.asarray(Xtr)
    Ytr = np.asarray(Ytr)
    Xte = np.asarray(Xte)
    Yte = np.asarray(Yte)
    
    metrics = ['accuracy', keras_precision, keras_recall, keras_auc,keras_f1, keras_fb]
    callbacks = [EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)]
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cvscores = []
    epochs = 1000
    if cv:
        with tqdm(total=10,desc='CV') as pbar:
            for train,test in kfold.split(Xtr,Ytr):
                model = SimpleMLP(input_dim=[N,M], embedding_dim=16, use_bn=True,use_dp=True)
                # Compile model
                model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=metrics)
                # Fit the model
                model.fit(Xtr[train], Ytr[train], epochs=epochs, batch_size=len(Ytr[train]), verbose=verbose, callbacks = callbacks)
                # evaluate the model
                scores = model.evaluate(Xtr[test], Ytr[test], verbose=0,batch_size=len(Ytr[test]))
                cvscores.append(scores)
                pbar.update(1)
                
            cvscores = np.asarray(cvscores)
            
        for i,n in enumerate(model.metrics_names):
            print(n,"%.2f (+/- %.2f)" % (np.mean(cvscores[:,i]), np.std(cvscores[:,i])))

    cvscores = []
    for i in tqdm(range(num_cl)):
        model = SimpleMLP(input_dim=[N,M], embedding_dim=16, use_bn=True,use_dp=True)
        model.compile('adagrad', loss='binary_crossentropy',metrics=metrics)
        model.fit(Xtr, Ytr, batch_size = len(Ytr), epochs = epochs, validation_split=0.0, shuffle=True, verbose = verbose, callbacks=callbacks)

        scores = model.evaluate(Xte, Yte, batch_size=len(Yte), verbose=0)
        cvscores.append(scores)
        
        p = model.predict(Xte)
        with open('./results/MLP/'+str(i)+'.txt', 'w') as f:
            for a,b in zip(Yte,p):
                f.write(str(a)+','+str(b[-1])+'\n')
        
    cvscores = np.asarray(cvscores)
    for i,n in enumerate(model.metrics_names):
        print(n,"%.2f (+/- %.2f)" % (np.mean(cvscores[:,i]), np.std(cvscores[:,i])))
        
    

if __name__ == '__main__':
    main(cv=True,verbose=0,num_cl=10)

