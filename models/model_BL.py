### Benchmark model using taxonomy nearest neighbor

from tqdm import tqdm
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,fbeta_score
from sklearn.model_selection import StratifiedKFold
from common import read_data
from matplotlib import colors
import matplotlib.pyplot as plt


class Node:
    def __init__(self, label):
        self.children = set()
        self.parents = set()
        self.label = label
        self.similarTo = defaultdict(float)
    
    def siblings(self):
        tmp = []
        for p in self.parents:
            tmp.extend(p.children())
        return tmp
    
    def __key__(self):
        return self.label
    
    def __hash__(self):
        return hash(self.label)
    
    def __eq__(self,other):
        return self.label == other.label
    
    def __lt__(self,other):
        return self.__key__() < other.__key__()
    
    def __str__(self):
        return self.__key__()
    
    
def sort_similarity(list_, adj):
    out = list_.copy()
    adj_copy = adj.copy()
    ci = 0
    seen = set()
    for i in range(len(list_)):
        seen.add(ci)
        for a in seen:
            adj_copy[ci,a] = -1
        ci = np.argmax(adj_copy[ci,:])
        out.insert(i, out.pop(ci))
        
    return out

def closest(c, adj, N = 10):
    out = [c]
    curr = c
    seen = set()
    adj_copy = adj.copy()
    while len(out) < N:
        seen.add(curr)
        for a in seen:
            adj_copy[curr,a] = -1
        curr = np.argmax(adj_copy[curr,:])
        out.append(curr)
    return out
    
class Model:
    def __init__(self, chemicals, taxons):
        self.chemicals = chemicals
        self.taxons = taxons
    
    def fit(self, X, Y):
        print('Fitting')
        train_effects = np.zeros((len(self.chemicals),len(self.taxons)))
        for x,y in zip(X,Y):
            c,t = x
            c,t = self.chemicals.index(c), self.taxons.index(t)
            train_effects[c,t] = y
            
        self.train_effects = train_effects
        
        N = len(self.chemicals)
        M = len(self.taxons)
        Pc_distance = np.zeros((N,N))
        Pt_distance = np.zeros((M,M))
        for i,c1 in enumerate(self.chemicals):
            for j, c2 in enumerate(self.chemicals):
                if i == j:
                    Pc_distance[i,j] = -1
                else:
                    Pc_distance[i,j] = c1.similarTo[c2]
        
        for i,c1 in enumerate(self.taxons):
            for j,c2 in enumerate(self.taxons):
                if i == j:
                    Pt_distance[i,j] = -1
                else:
                    Pt_distance[i,j] = 1/(distance(c1,c2)+1)
        
        self.Pc = Pc_distance
        self.Pt = Pt_distance
        
    def predict(self,X,depth=1,mode='alt'):
        # mode in ('alt','all')
        if isinstance(depth, int):
            depth = [depth]*2
            
        assert isinstance(depth, list)
        
        if mode == 'alt':
            return self.predict_alternating(X,depth)
        if mode == 'all':
            return self.predict_all(X,depth)
        else:
            raise NotImplementedError(mode+' Not Implemented')

    def predict_alternating(self, X, depth):
        N = max(depth)
        M = N
        C = self.chemicals.copy()
        T = self.taxons.copy()
        out = []
        C,T = sort_similarity(C,self.Pc),sort_similarity(T,self.Pt)
        for c,t in tqdm(X, desc='Predicting'):
            t = T.index(t)
            c = C.index(c)
            prediction = 0
            v1 = closest(t,self.Pt,M)
            v2 = closest(c,self.Pc,N)
            
            tmp1 = []
            tmp2 = []
            for a in v1:
                if not tmp1:
                    tmp1.append(a)
                else:
                    tmp1.extend([a]*2)
            for a in v2:
                tmp2.extend([a]*2)
            tmp2 = tmp2[:-1]
            
            for it, ic in zip(tmp1,tmp2):
                prediction = max(prediction, self.train_effects[ic,it])
        
            out.append(prediction)
        return out
    
    def predict_all(self, X, depth):
        N = depth[0]
        M = depth[1]
        C = self.chemicals.copy()
        T = self.taxons.copy()
        out = []
        C,T = sort_similarity(C,self.Pc),sort_similarity(T,self.Pt)
        for c,t in tqdm(X, desc='Predicting'):
            t = T.index(t)
            c = C.index(c)
            prediction = 0
            for nt in T[t:t+M]:
                for nc in C[c:c+N]:
                    it = self.taxons.index(nt)
                    ic = self.chemicals.index(nc)
                    prediction = max(prediction, self.train_effects[ic,it])
            
            out.append(prediction)
        return out

def is_vistited(x, path):
    return x in path

def path_from_root(node):
    if hasattr(node, 'path'):
        path = node.path
    else:
        queue = [node]
        path = []
        while queue:
            curr = queue.pop(0)
            path.append(curr)
            parents = list(curr.parents)
            queue.extend(parents)
            
            queue = list(filter(lambda x: x not in path, queue))
            
        path = path[::-1]
        node.path = path
   
    return path
    
def distance(node1, node2):
    path1 = path_from_root(node1) 
    path2 = path_from_root(node2)
    
    i=0
    while i<len(path1) and i<len(path2): 
        if path1[i] != path2[i]: 
            break
        i = i+1

    return (len(path1)+len(path2)-2*i) 

def main(cv=False):
    #LOADING DATA
    
    folder = './data/'
    chemical_graph = read_data(folder + 'chemical_graph.txt')
    chemical_similarity = read_data(folder + 'chemical_similarity.txt')
    taxonomy_graph = read_data(folder + 'taxonomy_graph.txt')
    
    chemicals = []
    for s,p,o,score in chemical_graph:
        tmp1 = Node(s)
        tmp2 = Node(o)
        if tmp1 in chemicals:
            idx = chemicals.index(tmp1)
            tmp1 = chemicals[idx]
            if tmp2 in chemicals:
                idx = chemicals.index(tmp2)
                tmp2 = chemicals[idx]
                tmp1.parents.add(tmp2)
            else:
                tmp1.parents.add(tmp2)
                chemicals.append(tmp2)
        else:
            if tmp2 in chemicals:
                idx = chemicals.index(tmp2)
                tmp2 = chemicals[idx]
                tmp1.parents.add(tmp2)
            else:
                tmp1.parents.add(tmp2)
                chemicals.append(tmp2)
            chemicals.append(tmp1)
               
    for s,p,o,score in chemical_similarity:
        tmp1 = Node(s)
        tmp2 = Node(o)
        try:
            idx1 = chemicals.index(tmp1)
            idx2 = chemicals.index(tmp2)
        except ValueError:
            continue
        
        tmp1 = chemicals[idx1]
        tmp2 = chemicals[idx2]
        
        tmp1.similarTo[tmp2] = score
        tmp2.similarTo[tmp1] = score
 
    taxons = list()
    for s,p,o,_ in taxonomy_graph:
        tmp1 = Node(s)
        tmp2 = Node(o)
        
        if tmp1 in taxons:
            idx = taxons.index(tmp1)
            tmp1 = taxons[idx]
            if tmp2 in taxons:
                idx = taxons.index(tmp2)
                tmp2 = taxons[idx]
                tmp1.parents.add(tmp2)
            else:
                tmp1.parents.add(tmp2)
                taxons.append(tmp2)
        else:
            if tmp2 in taxons:
                idx = taxons.index(tmp2)
                tmp2 = taxons[idx]
                tmp1.parents.add(tmp2)
            else:
                tmp1.parents.add(tmp2)
                taxons.append(tmp2)
            taxons.append(tmp1)
            
            
    Etr = read_data(folder + 'endpoints_train.txt')
    Ete = read_data(folder + 'endpoints_test.txt')
    
    C = set()
    T = set()
    Xtr = []
    Ytr = []
    Xte = []
    Yte = []
    
    for c,p,t,e in Etr:
        c = Node(c)
        t = Node(t)
        if c in chemicals and t in taxons:
            c = chemicals[chemicals.index(c)]
            t = taxons[taxons.index(t)]
        else:
            continue
        e = float(e)
        
        Xtr.append((c,t))
        Ytr.append(e)
        
        C.add(c)
        T.add(t)
        
    for c,p,t,e in Ete:
        c = Node(c)
        t = Node(t)
        if c in chemicals and t in taxons:
            c = chemicals[chemicals.index(c)]
            t = taxons[taxons.index(t)]
        else:
            continue
        e = float(e)
        
        Xte.append((c,t))
        Yte.append(e)
        
        C.add(c)
        T.add(t)
        
        
    ### MODELLING
    C = list(C)
    T = list(T)
    
    #CV
    metrics = [accuracy_score,precision_score,recall_score,f1_score,lambda x,y:fbeta_score(x,y,2)]
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cvscores = []
    if cv:
        for train,test in kfold.split(Xtr,Ytr):
            model = Model(C,T)
            model.fit([Xtr[t] for t in train],[Ytr[t] for t in train])
            y_hat = model.predict([Xtr[t] for t in test], depth = 30, mode='all')
            
            scores = [m([Ytr[t] for t in test],y_hat) for m in metrics]
            
            cvscores.append(scores)
        cvscores = np.asarray(cvscores)
        for i in range(len(metrics)):
            print(np.mean(cvscores[:,i]),np.std(cvscores[:,i]))
    
    #CL
    model = Model(C,T)
    model.fit(Xtr, Ytr)
    y_hat = model.predict(Xte, depth = 30, mode='all')
    scores = [m(Yte,y_hat) for m in metrics]
    print(d,scores)
    
if __name__ == '__main__':
    main(cv=False)
        
    
