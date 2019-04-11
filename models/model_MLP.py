### dense model

from keras import Model
from keras.layers import Dense, Dropout, BatchNormalization, Reshape, Embedding, Reshape, Concatenate
import numpy as np
from sklearn.model_selection import StratifiedKFold
from metrics import keras_auc,keras_precision,keras_recall,keras_f1,keras_fb
from common import prep, read_data
from keras.callbacks import EarlyStopping

class SimpleMLP(Model):

    def __init__(self, input_dim, embedding_dim = [128,128], use_bn=False, use_dp=False):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        
        if not isinstance(embedding_dim,list):
            embedding_dim = [embedding_dim] * 2
        
        self.e1 = Embedding(input_dim[0],embedding_dim[0])
        self.e2 = Embedding(input_dim[1],embedding_dim[1])
        self.m = Concatenate()
        
        self.l1 = Dense(32, activation='relu')
        
        self.ac = Dense(1, activation='sigmoid')
        self.r = Reshape((-1,))
        
        if self.use_dp:
            self.dp = Dropout(0.2)
        if self.use_bn:
            self.bn = BatchNormalization(axis=-1)

    def call(self, inputs):
        x1,x2 = inputs
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
        x = self.r(x)
        return x
    
def main(cv=False):
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
            e = float(e)
            Xtr.append(tmp)
            Ytr.append(e)
        except KeyError:
            pass
        
    for c,p,t,e in Ete:
        try:
            tmp = [mapping_c[c],mapping_t[t]]
            e = float(e)
            Xte.append(tmp)
            Yte.append(e)
        except KeyError:
            pass

    #Oversampling training
    u,c = np.unique(Ytr, return_counts=True)
    while c[0] != c[1]:
        idx = np.random.choice(len(Ytr))
        if Ytr[idx] == u[np.argmin(c)]:
            Ytr.append(Ytr[idx])
            Xtr.append(Xtr[idx])
        u,c = np.unique(Ytr, return_counts=True)

    Xtr = np.asarray(Xtr)
    Ytr = np.asarray(Ytr)
    Xte = np.asarray(Xte)
    Yte = np.asarray(Yte)

    X_tr_left, X_tr_right, Y_tr = prep(Xtr, Ytr)

    unique, counts = np.unique(Y_tr, return_counts=True)

    print("Training class distribution: {0} {1}".format(counts[0]/len(Y_tr),sum(counts[1:])/len(Y_tr)))
    metrics = ['accuracy', keras_precision, keras_recall, keras_auc,keras_f1, keras_fb]
    callbacks = [EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)]
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cvscores = []
    epochs = 1000
    if cv:
        for train,test in kfold.split(X_tr_left,Y_tr):
            model = SimpleMLP(input_dim=[N,M], embedding_dim=16, use_bn=True,use_dp=True)
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=metrics)
            # Fit the model
            model.fit([X_tr_left[train],X_tr_right[train]], Y_tr[train], epochs=epochs, batch_size=len(Y_tr[train]), verbose=0, callbacks = callbacks)
            # evaluate the model
            scores = model.evaluate([X_tr_left[test],X_tr_right[test]], Y_tr[test], verbose=0,batch_size=len(Y_tr[test]))
            cvscores.append(scores)
        cvscores = np.asarray(cvscores)
        for i,n in enumerate(model.metrics_names):
            print(n,"%.2f (+/- %.2f)" % (np.mean(cvscores[:,i]), np.std(cvscores[:,i])))

    model = SimpleMLP(input_dim=[N,M], embedding_dim=16, use_bn=True,use_dp=True)
    model.compile('adagrad', loss='binary_crossentropy',metrics=metrics)

    model.fit([X_tr_left,X_tr_right], Y_tr, batch_size = len(Y_tr), epochs = epochs, validation_split=0.0, shuffle=True, verbose = 2, callbacks=callbacks)

    X_te_left, X_te_right, Y_te = prep(Xte, Yte)

    unique, counts = np.unique(Y_te, return_counts=True)
    print("Test class distribution: {0} {1}".format(counts[0]/len(Y_te),counts[1]/len(Y_te)))

    r1 = model.evaluate([X_te_left,X_te_right], Y_te, batch_size=len(Y_te), verbose=0)
    for n,res in zip(model.metrics_names, r1):
        print(n,res)
    
    r2 = model.predict([X_te_left,X_te_right])
    y = Y_te
    
    return [r1,r2,y]
    

if __name__ == '__main__':
    main(cv=False)

