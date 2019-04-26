# utils.py

from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import URLError
from rdflib import Graph
import warnings
import pandas as pd

def prefixes(initNs):
    q = ''
    for k,i in initNs.items():
        q += "PREFIX\t"+k+':\t' + '<'+str(i)+'>\n'
    return q

def length_file(filename):
    return sum(1 for i in open(filename, 'rb'))

def read_data(filename):
    # Reading data line by line without header.
    
    with open(filename, 'r') as f:
        for l in f:
            cols = [a.strip() for a in l.split('|')]
            yield cols

def strip(string, symbols = ['#']):
    if not isinstance(string, str):
        return string
    if not isinstance(symbols,list):
        symbols = [symbols]
        
    tmp1 = string
    for s in symbols:
        tmp2 = string.split(s)[-1]
        if len(tmp2) < len(tmp1):
            tmp1 = tmp2
    return tmp1

def get_endpoints(filename):
    # get death rate from endpoint definitions.
    out = {}
    with open(filename, 'r') as f:
        for l in f:
            c,d = l.split('|')
            s = d.split()
            idx = [a for a in s if a.find('%') > -1]
            for i in idx:
                i = float(i.replace('%',''))
                out[c] = i/100
                break
            if len(idx) < 1:
                out[c] = -1
    return out
            
def test_endpoint(endpoint):
    sparql = SPARQLWrapper(endpoint)
    q = """
        SELECT ?s ?p ?o
        WHERE {?s ?p ?o}
        LIMIT 100
    """ 

    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return True
    except:
        return False
    
    
def query_endpoint(endpoint, q, var = 'p'):
    if not isinstance(var, list):
        var = [var]
        
    sparql = SPARQLWrapper(endpoint)
    
    out = {}

    try:
        sparql.setQuery(q)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        for v in var:
            try:
                out[v] = [r[v]['value'] for r in results['results']['bindings']]
            except KeyError:
                out[v] = [None] * len(results['results']['bindings'])
        return zip(*[out[k] for k in out])
    except:
        warnings.warn(q + '\n return an error. Possibly missformed query.')
        return []
        
    
def query_graph(graph, q):
    try:
        return list(graph.query(q))
    except:
        warnings.warn(q + '\n return an error. Possibly missformed query.')
        return []


def reverse_mapping(mapping):
    out = {}
    for k in mapping:
        if isinstance(mapping[k], str):
            out[mapping[k]] = k
        else:# isinstance(mapping[k], list) or isinstance(mapping[k], set):
            if len(mapping[k]) > 1:
                for i in mapping[k]:
                    out[i] = k
            elif len(mapping[k]) == 1:
                out[list(mapping[k])[0]] = k
            else:
                pass
        
    return out

