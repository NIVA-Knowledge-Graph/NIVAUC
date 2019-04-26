# chemid, inchikey, cas (no dash), ecotox data

import pandas as pd
from collections import defaultdict
import requests
from time import sleep

from tqdm import tqdm

from pubchempy import Compound, get_compounds, get_substances
from .utils import test_endpoint, prefixes, query_endpoint, query_graph, reverse_mapping, strip
import pubchempy

import warnings

from rdflib import Graph, Namespace, RDF
from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE

FORMAT = 'nt'

global over_air_cid2inchikey
over_air_cid2inchikey = {}
global over_air_inchikey2cid
over_air_inchikey2cid = {}

def chunks(l, n):
    if isinstance(l, set):
        l = list(l)
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def chemical_id2inchikey(chemical_file):
    chemical_data = pd.read_csv(chemical_file, sep='\t', dtype = str, usecols=['CHEMICAL_ID', 'INCHIKEY_STANDARD'])
    chemical_data = chemical_data.fillna(-1)
    out = {}
    for c,i in zip(chemical_data['CHEMICAL_ID'], chemical_data['INCHIKEY_STANDARD']):
        if all([c != -1, i != -1]):
            out[c] = i
    return out

def chemical_id2cas(chemical_file):
    chemical_data = pd.read_csv(chemical_file, sep='\t', dtype = str, usecols=['CHEMICAL_ID', 'CAS_NODASH'])
    chemical_data = chemical_data.fillna(-1)
    out = {}
    for c,i in zip(chemical_data['CHEMICAL_ID'], chemical_data['CAS_NODASH']):
        if all([c != -1, i != -1]):
            out[c] = i
    return out
    
def cas2inchikey(cas_file, chemical_file):
    
    chem2cas = chemical_id2cas(cas_file)
    chem2inchi = chemical_id2inchikey(chemical_file)
    
    out = {}
    for k,cas in chem2cas.items():
        out[cas] = chem2inchi[k]
    return out


def inchikey2cas():

    endpoint = 'https://query.wikidata.org/sparql'
    q = """
    SELECT DISTINCT ?compound ?compoundLabel ?inchikey ?cas WHERE {
    ?compound wdt:P31 wd:Q11173 .
    ?compound wdt:P235 ?inchikey .
    ?compound wdt:P231 ?cas .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    try:
        res = query_endpoint(endpoint, q, var = ['compound', 'compoundLabel', 'inchikey', 'cas'])
    except:
        return {}
    
    out = {}
    for _,_,inchikey,cas in res:
        out[inchikey] = cas.replace('-','')
    return out
        
def remove_namespace(cid, c = 'CID'):
    return cid.split(c)[-1]
    
def cid2inchikey(cids):
    out = {}
    for c in cids:
        if c in over_air_cid2inchikey:
            out[c] = over_air_cid2inchikey[c]
            continue
        sleep(0.2)
        try:
            out[c] = Compound.from_cid(c).to_dict(properties=['inchikey'])['inchikey']
            over_air_cid2inchikey[c] = out[c]
        except:
            out[c] = 'no mapping'
    return out

def inchikey2cid(ids):
    out = {}
    for c in ids:
        if c in over_air_inchikey2cid:
            out[c] = over_air_inchikey2cid[c]
            continue
        sleep(0.2)
        try:
            r = get_compounds(str(c), 'inchikey')
            r = r.pop()
            r = r.to_dict(properties=['cid'])
            out[c] = r['cid']
            over_air_inchikey2cid[c] = out[c]
        except:
            out[c] = 'no mapping'
    return out

def apply_mapping(ids, mapping):
    out = []
    
    for i in ids:
        try:
            a = mapping[i]
        except KeyError:
            a = 'no mapping'
        out.append(a)
    
    return out

def concat_dict(dict1, dict2):
    out = defaultdict(set)
    for k in dict1:
        out[k].add(dict1[k])
    for k in dict2:
        out[k].add(dict2[k])
    return out

def count_missing(res, from_, to_):
    try:
        idx = [i for i,x in enumerate(res) if x == 'set()']
        for i in idx:
            res[i] = 'no mapping'
    except ValueError:
        pass
    
    l1 = len(res)
    l2 = res.count('no mapping')
    
    if l2 > 0:
        warnings.warn(str(round(l2/l1,3))+' proportion of chemicals does not have mapping from '+from_ +' to '+to_)

def query_chembl(q, var):
    q = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX dbpedia2: <http://dbpedia.org/property/>
    PREFIX dbpedia: <http://dbpedia.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX cco: <http://rdf.ebi.ac.uk/terms/chembl#>
    """ + q
    endpoint = 'https://www.ebi.ac.uk/rdf/services/sparql'
    res = query_endpoint(endpoint, q, var)
    return res

def tanimoto(compound1, compound2):
    fp1 = int(compound1.fingerprint, 16)
    fp2 = int(compound2.fingerprint, 16)
    fp1_count = bin(fp1).count('1')
    fp2_count = bin(fp2).count('1')
    both_count = bin(fp1 & fp2).count('1')
    return float(both_count) / (fp1_count + fp2_count - both_count)

class Chemistry:
    """
    Loading and manipulating chemistry data from various sources.
    """
    def __init__(self, cas_file = None, chemical_file = None, directory = None, endpoint = '', cas_from_wikidata = False, verbose = False):
        """
        args:
            cas_file :: str
                path to cas to inchikey mapping file.
            chemical_file :: str    
                path to chemical_id to inchikey mapping file.
            directory :: str 
                optional, if loading from files.
            endpoint :: str 
                optional, host of endpoint with data.
                directory or endpoint must be present
            cas_from_wikidata :: boolean
                optional, whether to load cas numbers from wikidata in addtion to mapping from cas_file.
            verbose :: boolean
                optional, whether to display warnings.
        """
        assert directory or endpoint
        
        self.endpoint = endpoint
        self.verbose = verbose
        
        if not test_endpoint(endpoint):
            self.endpoint = ''
            if verbose: warnings.warn('Endpoint at:' + endpoint + ' not reached.')
        
        if directory:
            self.directory = directory
        else:
            self.directory = './'
        
            
        self.cas_file = cas_file
        self.chemical_file = chemical_file
        
        self.compound_namespace = Namespace('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/')
        self.vocab_namespace = Namespace('http://rdf.ncbi.nlm.nih.gov/pubchem/vocabulary#')
        self.obo_namespace = Namespace('http://purl.obolibrary.org/obo/')
        
        self.initNs = {'compound':self.compound_namespace, 'vocab':self.vocab_namespace, 'obo': self.obo_namespace}
        
        if not self.endpoint:
            self.graph = Graph()
            self._load_hierarchy()
            self._load_type()
        
        if not self.endpoint:
            for k,i in self.initNs.items():
                self.graph.bind(k,i)
            
        self._load_cas_mapping(load = True, from_wikidata = cas_from_wikidata)
        if self.chemical_file:
            self.chem2inchikey = chemical_id2inchikey(self.chemical_file)
        else:
            self.chem2inchikey = dict()
        self.cid2inchikey_mapping = dict()
        
    def __dict__(self):
        return {
                'namespace':self.namespace,
                'directory': self.directory,
                'endpoint': self.endpoint,
                'namespaces': self.initNs
            }
            
    def _apply_compound_namespace(self, cid):
        return self.compound_namespace['CID'+str(cid)]
    
    def _load_cas_mapping(self, load = True, from_wikidata = False):
        if load:
            try:
                df = pd.read_csv(self.directory+'inchikey_cas.csv',sep='|', dtype = str)
                self.cas_mapping = {k:i.split(',')[0] for k,i in zip(df['INCHIKEY'],df['CAS_NODASH'])}
                
            except FileNotFoundError:
                self._load_cas_mapping(load = False, from_wikidata = from_wikidata)
        else:
            if from_wikidata:
                tmp1 = inchikey2cas()
            else:
                tmp1 = {}
            if self.cas_file:
                tmp2 = cas2inchikey(self.cas_file, self.chemical_file)
            else:
                tmp2 = dict()
            tmp2 = reverse_mapping(tmp2)
            if len(tmp1) <= len(tmp2):
                if self.verbose: warnings.warn('Loading CAS from Wikidata failed or was disabled.')
                
            self.cas_mapping = concat_dict(tmp1, tmp2)
            for k in self.cas_mapping:
                self.cas_mapping[k] = list(self.cas_mapping[k])[0]
            col1 = []
            col2 = []
            for k,i in self.cas_mapping.items():
                col1.append(k)
                col2.append(i)
            if len(tmp1) > 0:
                df = pd.DataFrame({'INCHIKEY':col1,'CAS_NODASH':col2})
                df.to_csv(self.directory+'inchikey_cas.csv', sep='|')
    
    def _load_data(self, filename):
        g = Graph()
        g.parse(filename, format = 'ttl')
        return g
                
    def _load_hierarchy(self):
        self.graph += self._load_data(self.directory + 'pc_compound2parent.ttl')
        
    def _load_type(self):
        self.graph += self._load_data(self.directory + 'pc_compound_type.ttl')
    
    ### PRIVATE METHODS
    
    def _query(self, q, var):
        if self.endpoint:
            results = query_endpoint(self.endpoint, q, var = var)
        else:
            results = query_graph(self.graph, q)
        
        if len(var) < 2:
            results = [r[0] for r in results]
        
        return results
    
    def _similarity(self, cids):
        compounds = []
        for c in cids:
            sleep(0.2)
            try:
                c = Compound.from_cid(c)
                compounds.append(c)
            except pubchempy.NotFoundError as e:
                print(c,e)
            except pubchempy.BadRequestError as e:
                print(c,e)
                
        if self.verbose:
            a = 1 - len(compounds)/len(cids)
            if a > 0:
                warnings.warn('{0} proportion was not found during similarity calculation'.format(a))
        out = []
        for c1 in compounds:
            for c2 in compounds:
                out.append({'compound1':c1.cid, 'compound2':c2.cid, 'similarity':tanimoto(c1,c2)})
        return out
    
    def _all_compounds(self):
        # only returns 10000 elements from endpoint
        q = prefixes(self.initNs)
        q += """
            SELECT ?s {
            ?s  ?o  ?z
            FILTER (isURI(?s) && STRSTARTS(str(?s), str(compound:) ) )
            }
            """
        results = self.query(q, var = ['s'])
        if len(results) == 10000:
            warnings.warn('Only 10k compounds will be loaded from endpoint.')
        
        out = set()
        for row in results:
            out.add(row)
        return out
    
    def _get_parent(self, cid):
        q = prefixes(self.initNs)
        q += """
        SELECT ?p WHERE{
            <%s> vocab:has_parent ?p 
        }
        """ % cid
        results = self._query(q, var = ['p'])
        
        out = set()
        for row in results:
            out.add(row)
        return out
        
    def _get_child(self, cid):
        q = prefixes(self.initNs)
        q += """
        SELECT ?p WHERE {
            ?p vocab:has_parent <%s> 
        }
        """ % cid
        results = self._query(q, var = ['p'])
        
        out = set()
        for row in results:
            out.add(row)
        return out
    
    def _get_sibling(self, cid):
        q = prefixes(self.initNs)
        q += """
        SELECT ?p WHERE {
            <%s> vocab:has_parent [
                ^vocab:has_parent ?p
            ] .
        }
        """ % cid
        results = self._query(q, var = ['p'])
        
        out = set()
        for row in results:
            out.add(row)
        return out
        
    
    def _get_type(self, cid):
        q = prefixes(self.initNs)
        q += """
        SELECT ?p WHERE {
            <%s> rdf:type ?p 
        }
        """ % cid
        results = self._query(q, var = ['p'])
        
        out = set()
        for row in results:
            out.add(row)
        return out
    
    
    def _count_level(self, repeats, predicate):
        q = prefixes(self.initNs)
        predicate = '<' + predicate + '>'
        s = predicate
        for _ in range(repeats - 1):
            s += '/'
            s += predicate
            
        q += """
            ASK {[] %s []}
        """ % s
        
        if self.endpoint:
            sparql = SPARQLWrapper(self.endpoint)
            sparql.setQuery(q)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            results = results['boolean']
        else:
            results = self.hierarcy.query(q)
        
        return results
    
    def _count_levels(self, predicate):
        i = 1
        while self._count_level(i, predicate):
            i += 1
        return i
    
    def _subset(self, cids, levels = 1, down = False):
        if levels == 0:
            return cids
        
        if levels == -1:
            levels = float('inf')
        
        if down:
            f = self._get_child
        else:
            f = self._get_parent
        
        out = set.union(*[set(f(c)) for c in cids])
        tmp = out
        while tmp and levels > 0:
            tmp = set.union(*[set(f(c)) for c in tmp])
            out |= tmp
            levels -= 1
            
        return out
    
    def _get_features(self, cids, params = None):
        out = {}
        for c in cids:
            sleep(0.2)
            try:
                if params:
                    out[c] = Compound.from_cid(c).to_dict(properties = params)
                else:
                    out[c] = Compound.from_cid(c).to_dict()
            except pubchempy.NotFoundError as e:
                print(c,e)
            except pubchempy.BadRequestError as e:
                print(c,e)
                
        return out
    
    def _save(self, directory = './'):
        if self.endpoint:
            raise NotImplementedError ('save from endpoint is not implemented.')
        else: 
            self.graph.serialize(directory + 'chemistry.' + FORMAT , format = FORMAT)
    
    def _get_chembl_superclasses(self, uris):
        res = []
        for ur in chunks(uris, 10):
            s = ' '.join(['<'+u+'>' for u in ur])
            q = """ SELECT ?s ?c {
                    VALUES ?s { %s }
                    ?s rdfs:subClassOf | rdf:type ?c .
                    FILTER (!isBlank(?c))
                }
            """ % s
            res.extend(query_chembl(q, var = ['s', 'c']))
        return res
    
    def _get_chembl_siblings(self, uris):
        res = []
        for ur in chunks(uris, 10):
            s = ' '.join(['<'+u+'>' for u in ur])
            q = """ SELECT ?s ?c {
                    VALUES ?s { %s }
                    ?s rdfs:subClassOf | rdf:type [ 
                    ^rdfs:subClassOf | ^rdf:type ?c 
                    ] .
                }
            """ % s
            res.extend(query_chembl(q, var = ['s', 'c']))
        return res
        
    
    ### PUBLIC FUNCTIONS
    
    def similarity(self, cids, from_ = None, to_ = None):
        """
        Gets parents of input ids.
        Args:
            cids :: list
                list of ids. Must be of type cid.
            from_ :: str
                optional, input type, inchikey, cas, chemical_id, or cid.
            to_ :: str
                optional, output type, inchikey, cas, chemical_id, or cid.
        Returns: 
            out :: list 
                dict:
                    key - compound1, compound2, similarity
                    item - cid, cid, float
        """
        
        def f(i, mapping):
            if isinstance(i, float):
                return i
            try:
                return mapping[str(i)]
            except KeyError:
                return 'no mapping'
        
        if from_: 
            tmp = self.convert_ids(from_, 'cid', cids)
            cids = [c for c in tmp if c != 'no mapping']
        out = self._similarity(cids)
        if to_:
            mapping = {k:i for k,i in zip(cids, self.convert_ids('cid', to_, cids))}
            out = [{k:f(d[k],mapping) for k in d} for d in out]
        return out
    
    
    def save(self, directory = './'):
        """
        Save data to RDF.
        Args:
            directory :: str 
                path to save files at.
        """
        self._save(directory)
    
    def get_parents(self, cids, from_ = None, to_ = None):
        """
        Gets parents of input ids.
        Args:
            cids :: list
                list of ids. Must be of type cid with appropriate rdf namespace. 
            from_ :: str
                optional, input type, inchikey, cas, chemical_id, or cid.
            to_ :: str
                optional, output type, inchikey, cas, chemical_id, or cid.
        Returns: 
            out :: dict 
                key - id
                item - list of parents of id
        """
        if from_: 
            cids = self.convert_ids(from_, 'cid', cids)
            cids = [c for c in cids if c != 'no mapping']
            cids = [self._apply_compound_namespace(c) for c in cids]
        
        out = {c:list(self._get_parent(c)) for c in cids}
        if to_:
            cids = set([strip(c,['#', '/', 'CID']) for c in cids])
            for k in out:
                cids |= set([strip(c,['#', '/', 'CID']) for c in out[k]])
            mapping = {k:i for k,i in zip(cids, self.convert_ids('cid', to_, cids))}
            out = {mapping[strip(k,['#', '/', 'CID'])]:[mapping[strip(a,['#', '/', 'CID'])] for a in out[k]] for k in out}
        return out
    
    def get_children(self, cids, from_ = None, to_ = None):
        """
        Gets children of input ids.
        Args:
            cids :: list
                list of ids. Must be of type cid with appropriate rdf namespace. 
            from_ :: str
                optional, input type, inchikey, cas, chemical_id, or cid.
            to_ :: str
                optional, output type, inchikey, cas, chemical_id, or cid.
        Returns: 
            out :: dict 
                key - id
                item - list of children of id
        """
        if from_: 
            cids = self.convert_ids(from_, 'cid', cids)
            cids = [c for c in cids if c != 'no mapping']
            cids = [self._apply_compound_namespace(c) for c in cids]
        
        out = {c:list(self._get_child(c)) for c in cids}
        if to_:
            cids = set([strip(c,['#', '/', 'CID']) for c in cids])
            for k in out:
                cids |= set([strip(c,['#', '/', 'CID']) for c in out[k]])
            mapping = {k:i for k,i in zip(cids, self.convert_ids('cid', to_, cids))}
            out = {mapping[strip(k,['#', '/', 'CID'])]:[mapping[strip(a,['#', '/', 'CID'])] for a in out[k]] for k in out}
        return out
    
    def get_siblings(self, cids, from_ = None, to_ = None):
        """
        Gets siblings of input ids. i.e. children of parent.
        Args:
            cids :: list
                list of ids. Must be of type cid with appropriate rdf namespace. 
            from_ :: str
                optional, input type, inchikey, cas, chemical_id, or cid.
            to_ :: str
                optional, output type, inchikey, cas, chemical_id, or cid.
        Returns: 
            out :: dict 
                key - id
                item - list of siblings of id
        """
        if from_: 
            cids = self.convert_ids(from_, 'cid', cids)
            cids = [c for c in cids if c != 'no mapping']
            cids = [self._apply_compound_namespace(c) for c in cids]
            
        out =  {c:self._get_sibling(c) for c in cids}
        if to_:
            cids = set([strip(c,['#', '/', 'CID']) for c in cids])
            for k in out:
                cids |= set([strip(c,['#', '/', 'CID']) for c in out[k]])
            mapping = {k:i for k,i in zip(cids, self.convert_ids('cid', to_, cids))}
            out = {mapping[strip(k,['#', '/', 'CID'])]:[mapping[strip(a,['#', '/', 'CID'])] for a in out[k]] for k in out}
        return out
        
        
    def get_types(self, cids, from_ = None, to_ = None):
        """
        Gets types (classes) of input ids.
        Args:
            cids :: list
                list of ids. Must be of type cid with appropriate rdf namespace. 
            from_ :: str
                optional, input type, inchikey, cas, chemical_id, or cid.
            to_ :: str
                optional, output type, inchikey, cas, chemical_id, or cid.
        Returns: 
            out :: dict 
                key - id
                item - list of types of id
        """
        def apply_mapping(a, mapping):
            try: 
                return mapping[a]
            except:
                return a
        
        if from_: 
            cids = self.convert_ids(from_, 'cid', cids)
            cids = [c for c in cids if c != 'no mapping']
            cids = [self._apply_compound_namespace(c) for c in cids]
        
        out =  {c:self._get_type(c) for c in cids}
        if to_:
            cids = [strip(c,['#', '/', 'CID']) for c in cids]
            mapping = {k:i for k,i in zip(cids, self.convert_ids('cid', to_, cids))}
            out = {apply_mapping(k,mapping):set([apply_mapping(a,mapping) for a in out[k]]) for k in out}
        return out
        
    def get_chembl_superclasses(self, uris):
        res = self._get_chembl_superclasses(uris)
        out = defaultdict(set)
        for s,c in res:
            out[s].add(c)
        return out
        
    def subset(self, cids, up_levels = 1, down_levels = 1, from_ = None, to_ = None):
        """
        Traverses tree up/down and returns all entities passed.
        Args:
            cids :: list
                list of ids. Must be of type cid with appropriate rdf namespace. 
            up_levels :: int
                number of levels to traverse tree upward. -1 denotes to the top.
            down_levels :: int
                number of levels to traverse tree downward. -1 denotes to the bottom.
            from_ :: str
                optional, input type, inchikey, cas, chemical_id, or cid.
            to_ :: str
                optional, output type, inchikey, cas, chemical_id, or cid.
        Returns: 
            out :: list 
                all entities passed while traversing.
        """
        
        if from_: 
            cids = self.convert_ids(from_, 'cid', cids)
            cids = [c for c in cids if c != 'no mapping']
            cids = [self._apply_compound_namespace(c) for c in cids]
        
        cids = set(cids)
        
        tmp1 = self._subset(cids, up_levels, down = False)
        tmp2 = self._subset(cids, down_levels, down = True)
        tmp = cids | tmp1 | tmp2
        for k,i in self.get_siblings(tmp1).items():
            tmp |= i
        for k,i in self.get_siblings(tmp2).items():
            tmp |= i
        
        if to_:
            tmp = [strip(c, ['CID']) for c in tmp]
            tmp = self.convert_ids('cid', to_, list(tmp))
            
        return tmp
    
    def compounds(self):
        """
        Returns all compounts.
        Returns:
            list
        """
        return self._all_compounds()
    
    def construct_tree(self, subset = None, from_ = None, to_ = None):
        """
        Traverses tree up and down and returns all entities passed.
        Args:
            cids :: list
                list of ids. Must be of type cid with appropriate rdf namespace. 
            up_levels :: int
                number of levels to traverse tree upward. -1 denotes to the top.
            down_levels :: int
                number of levels to traverse tree downward. -1 denotes to the bottom.
            from_ :: str
                optional, input type, inchikey, cas, chemical_id, or cid.
            to_ :: str
                optional, output type, inchikey, cas, chemical_id, or cid.
        Returns: 
            out :: list 
                all entities passed while traversing.
        """
        if from_: 
            subset = self.convert_ids(from_, 'cid', subset)
            subset = [c for c in subset if c != 'no mapping']
            subset = [self._apply_compound_namespace(c) for c in subset]
            
        if not subset:
            subset = self._all_compounds()
        
        out = []
        all_keys = set()
        for k,i in self.get_parents(subset).items():
            for a in i:
                out.append({'child':k, 'parent':a, 'score':1})
                all_keys.add(strip(a,'CID'))
                all_keys.add(strip(k,'CID'))
            
        for k,i in self.get_children(subset).items():
            for a in i:
                out.append({'child':a, 'parent':k, 'score':1})
                all_keys.add(strip(a,'CID'))
                all_keys.add(strip(k,'CID'))
        
   
        def f(i, mapping):
            if isinstance(i, int):
                return i
            return mapping[i]
        
        if to_:
            mapping = {k:i for k,i in zip(list(all_keys), self.convert_ids('cid', to_, list(all_keys)))}
            out = [{k:f(strip(d[k],'CID'), mapping) for k in d} for d in out]
            
        return out
    
    def query(self, q, var):
        """
        Custom query of the data.
        Args:
            q :: str 
                query, without namespace prefixes, self.prefixes_used() will provide namespaces.
            var :: list
                list of variables to extract from results.
        Return:
            list of tuples (of len(var))
        """
        q = prefixes(self.initNs) + q
        return self._query(q, var = var)
    
    def prefixes_used(self):
        return self.initNs
    
    def get_features(self, cids, from_ = None, to_ = None, params = None):
        """
        Get features for chemicals.  
        Args:
            cids :: list
                list of ids. Must be of type cid with appropriate rdf namespace. 
            from_ :: str
                optional, input type, inchikey, cas, chemical_id, or cid.
            to_ :: str
                optional, output type, inchikey, cas, chemical_id, or cid.
        Returns: 
            out :: dict
                key - id
                item - dict 
                    key - feature 
                    item - value
        """
        if from_: 
            cids = self.convert_ids(from_,'cid', cids)
        
        out = self._get_features(cids, params)
        
        if to_:
            f = {k:i for k,i in zip(list(out.keys()), self.convert_ids('cid', to_, list(out.keys())))}
            out = {f[k]:out[k] for k in out}
        
        return out
    
    def class_hierarchy(self, cids, from_ = None, to_ = None):
        """
        Construct class hierarchy on adjecency form.
        Args:
            cids :: list
                list of ids. Must be of type cid with appropriate rdf namespace. 
            from_ :: str
                optional, input type, inchikey, cas, chemical_id, or cid.
            to_ :: str
                optional, output type, inchikey, cas, chemical_id, or cid.
        Returns:
            out :: list
                list of child, parent, score pairs
        """
        if from_:
            cids = self.convert_ids(from_, 'cid', cids)
            cids = [self._apply_compound_namespace(c) for c in cids if c != 'no mapping']
        
        types = self.get_types(cids)
        
        out = set()
        for k in types:
            for i in types[k]:
                out.add((str(k), str(i), 1))
        
        tmp = set([p for _,p,_ in out])
        
        while tmp:
            l1 = len(out)
            tmp = tmp - set.union(*[set([a,b]) for a,b,_ in out])
            classes = self.get_chembl_superclasses(list(tmp))
            tmp = set()
            for k in classes:
                for i in classes[k]:
                    tmp.add(str(i))
                    out.add((str(k), str(i), 1))
            l2 = len(out)
            if l2 <= l1:
                break
        
        return out
        
            
    def convert_ids(self, from_, to_, ids):
        """
        Convert between types of ids used in chemical data.
        
        Args:
            from_ :: str 
                input id type, inchikey, cas, chemical_id or cid.
            to_ :: str 
                output id type, inchikey, cas, chemical_id or cid.
            ids :: list
                list of ids, 'no mapping' if no mapping between from_ and to_ exists.
        Return:
            ids :: list
                converted ids.
        Raises:
            NotImplementedError: if from_ or to_ not in (inchikey, cas, chemical_id, cid)
        """
        ids = [str(i) for i in ids]
        l1 = len(ids)
        
        if len(ids) < 1:
            return ids
        
        if from_ == to_:
            return ids
        
        elif from_ == 'inchikey':
            if to_ == 'cas':
                mapping = self.cas_mapping
            elif to_ == 'chemical_id':
                mapping = reverse_mapping(self.chem2inchikey)
            elif to_ == 'cid':
                mapping = inchikey2cid(ids)
            else:
                raise NotImplementedError (from_+' to ' + to_ + ' is not supported') 
            
        elif from_ == 'cas':
            if to_ == 'inchikey':
                mapping = reverse_mapping(self.cas_mapping)
            elif to_ == 'chemical_id':
                mapping = reverse_mapping(self.cas_mapping)
                ids = apply_mapping(ids, mapping)
                if self.verbose: count_missing(ids, from_, to_)
                return self.convert_ids('inchikey', 'chemical_id', ids)
            elif to_ == 'cid':
                mapping = reverse_mapping(self.cas_mapping)
                ids = apply_mapping(ids, mapping)
                if self.verbose: count_missing(ids, from_, to_)
                return self.convert_ids('inchikey', 'cid', ids)
            else:
                raise NotImplementedError (from_+' to ' + to_ + ' is not supported') 
            
        elif from_ == 'chemical_id':
            if to_ == 'inchikey':
                mapping = self.chem2inchikey
            elif to_ == 'cas':
                mapping = self.chem2inchikey
                ids = apply_mapping(ids, mapping)
                if self.verbose: count_missing(ids, from_, to_)
                return self.convert_ids('inchikey', 'cas', ids)
            elif to_ == 'cid':
                mapping = self.chem2inchikey
                ids = apply_mapping(ids, mapping)
                if self.verbose: count_missing(ids, from_, to_)
                return self.convert_ids('inchikey', 'cid', ids)
            else:
                raise NotImplementedError (from_+' to ' + to_ + ' is not supported') 
        
        elif from_ == 'cid':
            if to_ == 'inchikey':
                mapping = cid2inchikey(ids)
            elif to_ == 'cas':
                mapping = cid2inchikey(ids)
                ids = apply_mapping(ids, mapping)
                if self.verbose: count_missing(ids, from_, to_)
                return self.convert_ids('inchikey', 'cas', ids)
            elif to_ == 'chemical_id':
                mapping = cid2inchikey(ids)
                ids = apply_mapping(ids, mapping)
                if self.verbose: count_missing(ids, from_, to_)
                return self.convert_ids('inchikey', 'chemical_id', ids)
            else: 
                raise NotImplementedError (from_+' to ' + to_ + ' is not supported')
        
        else:
            raise NotImplementedError (from_+' to ' + to_ + ' is not supported')
        
        
        res = apply_mapping(ids, mapping)
        res = [str(r) for r in res]
        
        if self.verbose: count_missing(res, from_, to_)
        
        return res
        
                
                
