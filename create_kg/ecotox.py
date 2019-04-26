#ecotox

from collections import defaultdict
import pandas as pd
import os
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import FOAF, RDF, OWL, RDFS
from rdflib.plugins.sparql import prepareQuery
from .utils import get_endpoints, test_endpoint, prefixes, query_endpoint, query_graph, reverse_mapping, strip
import re

from .keys import get_time_units
time_units = get_time_units()

import itertools
import time
from tqdm import tqdm

from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE

nan_values = ['nan', float('nan'),'--','-X','NA','NC',-1,'','sp.', -1,'sp,','var.','variant']

numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)

EXT = '.utf8.converted'
FORMAT = 'nt'


def get_wikidata_name(ids):
    
    s = ','.join(['\"'+ str(i) +'\"' for i in ids])
    
    endpoint = 'https://query.wikidata.org/sparql'
    q = """
    SELECT ?taxid ?name WHERE {
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    ?item wdt:P685 ?taxid . 
    ?item wdt:P225 ?name .
    FILTER (?taxid in ( %s ))
    }
    """ %s
    
    try:
        res = query_endpoint(endpoint, q, var = ['taxid','name'])
    except:
        return []
    
    out = []
    for i in res:
        out.append(i)
    return out

class Ecotox:
    """
    Provides easy access to ecotox data.
    """
    def __init__(self, directory = '', endpoint = '', namespace = 'http://example.org/ecotox', endpoints_to_consider = None):
        """
        args:
            directory :: str 
                data directory, must contain results and tests files as well as validation subdirectory.
            endpoint :: str
                sparql endpoint. If endpoint is provided, files will not be loaded.
            namespace :: str   
                base namespace for the dataset.
        """
        if not test_endpoint(endpoint):
            self.endpoint = ''
        self.directory = directory
        self.results_file = directory + '/results' + EXT
        self.tests_file = directory + '/tests' + EXT
        self.species_file = directory + '/validation/species' + EXT
        self.species_synonyms_file = directory + '/validation/species_synonyms' + EXT
        self.endpoints_file = directory + '/validation/endpoint_codes' + EXT
        self.chemical_name_file = directory + '/validation/chemicals' + EXT
        self.units_file = directory+'/RA_UNIT_CONVERSIONS.tsv'
        
        self.namespace = Namespace(namespace+'/')
        self.onto_namespace = Namespace(namespace+'#')
        self.endpoint = endpoint
        self.initNs = {'rdf':RDF, 'foaf':FOAF, 'ns':self.namespace, 'nso':self.onto_namespace, 'owl':OWL, 'rdfs':RDFS}
        
        self.endpoints_to_consider = endpoints_to_consider
        
        if not endpoint:
            self.graph = Graph()
            self._load_units_convertion()
            self._load_data()
            self._load_chemicals()
            self._load_species()
            self._load_data_graph()
            self._load_hierarchy()
            self._sanity_checks()
        
        if not self.endpoint:
            for k,i in self.initNs.items():
                self.graph.bind(k,i)
    
    def __dict__(self):
        return {
                'namespace':self.namespace,
                'directory': self.directory,
                'endpoint': self.endpoint,
                'namespaces': self.initNs
            }
    
    def _sanity_checks(self):
        pass
    
    def _load_units_convertion(self):
        try:
            df = pd.read_csv(self.units_file, sep='\t')
            units = defaultdict(lambda:1)
            for k,f,mf in zip(df['ORIGINAL_UNIT'],df['FACTOR'],df['MATRIX_FACTOR']):
                factor = 1
                try:
                    f = float(f)
                    if f != 0 and f:
                        factor *= f
                except:
                    pass
                
                try:
                    mf = float(mf)
                    if f != 0 and f:
                        factor *= f
                except:
                    pass
                    
                units[k] = factor
                
        except FileNotFoundError:
            units = defaultdict(lambda:1)
        
        self.units = units
        
    def _load_data(self):
        endpoints = get_endpoints(self.endpoints_file)
        # description of experiments, includes cas(chemical_id), species, duration.
        tests = pd.read_csv(self.tests_file, sep='|', low_memory=False, dtype = str).fillna(-1)
        data = {}
        with tqdm(total = len(tests['test_id']), desc='Loading tests data') as pbar:
            for test_id, cas_number, species_number, stdm, stdu in zip(tests['test_id'],tests['test_cas'],tests['species_number'],tests['study_duration_mean'],tests['study_duration_unit']):
                pbar.update(1)
                if any([test_id in nan_values, cas_number in nan_values, species_number in nan_values]):
                    continue
                try: 
                    unit = time_units[stdu]
                except KeyError:
                    unit = -1
                
                data[test_id] = {'cas': cas_number, 'species': species_number}

        # description of experiment results, includes endpoints, concentration.
        remove_from_conc = set()
        results_data = {}
        results = pd.read_csv(self.results_file, sep='|', low_memory=False, dtype = str).fillna(-1)
        with tqdm(total = len(results['test_id']), desc='Loading endpoints data') as pbar:
            for test_id, endpoint, conc, conc_unit in zip(results['test_id'], results['endpoint'], results['conc1_mean'], results['conc1_unit']):
                pbar.update(1)
                if endpoint in nan_values:
                    continue
                if self.endpoints_to_consider:
                    if not any([a in endpoint for a in self.endpoints_to_consider]):
                        continue
                try:
                    if not conc in nan_values:
                        conc = rx.findall(conc)
                        if len(conc) < 1:
                            conc = -1
                        else:
                            conc = conc.pop(0)
                    else:
                        conc = -1
                    
                    tmp = data[test_id]
                    tmp['endpoint'] = endpoints[endpoint] #death rate
                    tmp['endpoint_label'] = endpoint 
                    tmp['concentration'] = float(conc)
                    tmp['concentration_unit'] = conc_unit
                    results_data[test_id] = tmp
                except KeyError:
                    pass

        self.data = results_data
        
    def _load_data_graph(self):
        g = Graph()
        with tqdm(total = len(self.data), desc='Converting endpoints to graph') as pbar:
            for k in self.data:
                pbar.update(1)
                c,s,e,conc,concunit,el = self.data[k]['cas'], self.data[k]['species'], self.data[k]['endpoint'], self.data[k]['concentration'], self.data[k]['concentration_unit'],self.data[k]['endpoint_label']
                
                s = self.namespace['taxon/'+str(s)]
                c = self.namespace['chemical/'+str(c)]
                g.add((s, RDF.type, self.onto_namespace['Taxon']))
                g.add((c, RDF.type, self.onto_namespace['Chemical']))
                b = BNode()
                g.add((b, self.onto_namespace['species'], s))
                g.add((b, self.onto_namespace['endpoint'], Literal(float(e))))
                g.add((b, self.onto_namespace['concentration'], Literal(float(conc))))
                g.add((b, self.onto_namespace['concentration_unit'], Literal(str(concunit))))
                g.add((b, self.onto_namespace['endpoint_label'], Literal(str(el))))
                g.add((c, self.onto_namespace['effects'], b))
            
        self.graph += g
    
    def _load_species(self):
        # creates a set of names for each species id. 
        
        g = Graph()
        
        species = defaultdict(set)
        df1 = pd.read_csv(self.species_file, sep='|', low_memory=False, dtype = str).fillna(-1)
        df2 = pd.read_csv(self.species_synonyms_file, sep='|', low_memory=False, dtype = str).fillna(-1)
        with tqdm(total = len(df1['species_number']), desc='Loading species') as pbar:
            for s, cn, ln, group in zip(df1['species_number'], df1['common_name'], df1['latin_name'],df1['ecotox_group']):
                pbar.update(1)
                if s in nan_values:
                    continue
                s = self.namespace['taxon/'+s]
                if not group in nan_values:
                    names = group.split(',')
                    group = group.replace('/','')
                    group = group.replace('.','')
                    group = group.replace(' ','')
                    tmp = group.split(',')
                    group_uri = [self.namespace['group/'+gr] for gr in tmp]
                    
                    for gri,n in zip(group_uri,names):
                        g.add((s, RDFS.subClassOf, gri))
                        g.add((gri, RDFS.label, Literal(n)))
                        g.add((gri, RDF.type, self.onto_namespace['Group']))
                    
                g.add((s, RDF.type, self.onto_namespace['Taxon']))
                    
                if not cn in nan_values:
                    g.add((s, self.onto_namespace['commonName'], Literal(cn)))
                if not ln in nan_values:
                    g.add((s, self.onto_namespace['latinName'], Literal(ln)))
            
            g.add((self.onto_namespace['commonName'], RDFS.subPropertyOf, RDFS.label))
            g.add((self.onto_namespace['latinName'], RDFS.subPropertyOf, RDFS.label))
        
        self.graph += g
    
    def _load_chemicals(self):
        # all chemical ids in dataset.
        tests = pd.read_csv(self.chemical_name_file, sep='|', low_memory=False, dtype = str).fillna(-1)
       
        g = Graph()
        with tqdm(total = len(tests['cas_number']), desc='Loading chemicals') as pbar:
            for c, n in zip(tests['cas_number'], tests['chemical_name']):
                pbar.update(1)
                if not any([c in nan_values, n in nan_values]):
                    c = self.namespace['chemical/'+str(c)]
                    g.add((c, RDF.type, self.onto_namespace['Chemical']))
                    if n:
                        g.add((c, RDFS.label, Literal(n)))
                
        self.graph += g
        
    def _load_hierarchy(self):
        
        g = Graph()
        tmp = set()
        df = pd.read_csv(self.species_file, sep= '|', low_memory = False, dtype = str).fillna(-1)
        ranks = ['variety','subspecies','species','genus','family','tax_order','class','superclass','subphylum_div','phylum_division','kingdom']
        #ranks = ranks[::-1]
        with tqdm(total = len(df['species_number']), desc='Loading hierarchy') as pbar:
            for sn, ln, lineage in zip(df['species_number'],df['latin_name'], zip(df['variety'],df['subspecies'],df['species'],df['genus'],df['family'],df['tax_order'],df['class'],df['superclass'],df['subphylum_div'],df['phylum_division'],df['kingdom'])):
                pbar.update(1)
                curr = self.namespace['taxon/'+sn]
                g.add((curr, RDF.type, self.onto_namespace['Taxon']))
                for i, a in enumerate(lineage):
                    tmp.add(curr)
                    if not a in nan_values:
                        n = a
                        l = set(self.graph.subjects(predicate=RDFS.label,object=Literal(n))) & tmp
                        try:
                            a = l.pop()
                            self.add_same_as_species(l)
                        except KeyError:
                            a = a.replace(' ','_')
                            a = self.namespace['taxon/'+a]
                            g.add((a, RDFS.label, Literal(n)))
                            g.add((a, RDF.type, self.onto_namespace['Taxon']))
                            r = self.namespace['rank/'+ranks[i]]
                            g.add((a, self.onto_namespace['rank'], r))
                            g.add((r, RDFS.label, Literal(ranks[i])))
                            g.add((r, RDF.type, self.onto_namespace['Rank']))
                        g.add((curr, RDFS.subClassOf, a))
                        curr = a
                
        self.graph += g
        
    ### PRIVATE FUNCTIONS
        
    def _query(self, q, var):
        if self.endpoint:
            results = query_endpoint(self.endpoint, q, var = var)
        else:
            results = query_graph(self.graph, q)
        
        if len(var) < 2:
            results = [r[0] for r in results]
        
        return results
        
        
    def _get_parent(self, taxon):
        q = prefixes(self.initNs)
        q += """
            SELECT ?p WHERE {
                <%s> rdf:type | rdfs:subClassOf ?p .
            }
        """ % taxon
        out = self._query(q, ['p'])
        return out
    
    def _lineage(self, taxon, depth):
        out = []
        p = self._get_parent(taxon)
        
        if depth > 10 or not p:
            return out
        
        if taxon in p:
            p.remove(taxon)
        if len(p) == 1:
            out.append({'child':taxon, 'parent': p[0], 'score':1})
        else:
            for s in p:
                out.append({'child':taxon, 'parent': s, 'score':1})
                out.extend(self._lineage(s, depth = depth + 1))
        
        return out
    
    def _hierarchy(self, species):
        out = []
            
        for curr in species:
            try:
                out.extend(self._lineage(curr, depth = 0))
            except RecursionError:
                pass
        
        out = [a for a in out if a]
        return out
    
    
    def _chemicals(self):
        # return all chemicals
        sparql = SPARQLWrapper(self.endpoint)
        q = prefixes(self.initNs)
        q += """
            SELECT DISTINCT ?c WHERE {
                ?c rdf:type nso:Chemical
            }
        """
        results = self._query(q, ['c'])
            
        out = set()
        for row in results:
            out.add(row)
        return out
    
    def _species(self):
        # return all species
        sparql = SPARQLWrapper(self.endpoint)
        q = prefixes(self.initNs)
        q += """
            SELECT DISTINCT ?c WHERE {
                ?c rdf:type nso:Taxon
            }
        """
        results = self._query(q, ['c'])
        out = set()
        for row in results:
            out.add(row)
        return out
    
    def _species_names(self, subset = None):
        # return names of species.
        sparql = SPARQLWrapper(self.endpoint)
        initNs = {'rdf':RDF, 'foaf':FOAF, 'ns':self.namespace}
        q = prefixes(initNs)
        if subset:
            q += """
                SELECT ?c ?n {
                    values ?c { %s }
                    ?c rdf:type nso:Taxon ;
                        rdfs:label ?n .
                }
            """ % '{0}'.format(' '.join(['<'+str(s)+'>' for s in subset]))
        else:
            q += """
            SELECT ?c ?n WHERE {
                ?c rdf:type nso:Taxon ;
                    rdfs:label ?n .
            }
        """
        
        results = self._query(q, ['c','n'])
            
        out = defaultdict(set)
        for s,n in results:
            out[s].add(str(n))
        return out
    
    def _chemical_names(self, subset = None):
        # return names of species.
        sparql = SPARQLWrapper(self.endpoint)
        initNs = {'rdf':RDF, 'foaf':FOAF, 'ns':self.namespace}
        q = prefixes(initNs)
        if subset:
            q += """
                SELECT ?c ?n {
                    values ?c { %s }
                    ?c rdf:type nso:Chemical ;
                        rdfs:label ?n .
                }
            """ % '{0}'.format(' '.join(['<'+str(s)+'>' for s in subset]))
        else:
            q += """
            SELECT ?c ?n WHERE {
                ?c rdf:type nso:Chemical ;
                    rdfs:label ?n .
            }
        """
        
        results = self._query(q, ['c','n'])
            
        out = defaultdict(set)
        for s,n in results:
            out[s].add(str(n))
        return out
    
    def _endpoint(self, c, s):
        q = prefixes(self.initNs)
        q += """
            SELECT ?e ?cc ?cu ?el WHERE {
                <%s> nso:effects ?c .
                ?c nso:species <%s> .
               ?c nso:endpoint ?e .
               ?c nso:concentration ?cc .
               ?c nso:concentration_unit ?cu .
               ?c nso:endpoint_label ?el .
            }""" % (str(c), str(s))
        return self._query(q, ['e','cc','cu','el'])
    
    def _all_endpoints(self):
        out = []
        for k in self.data:
            c,s,e,conc,concunit,el = self.data[k]['cas'], self.data[k]['species'], self.data[k]['endpoint'], self.data[k]['concentration'], self.data[k]['concentration_unit'],self.data[k]['endpoint_label']
            
            out.append({'cas':c,'taxon':s,'endpoint':e,'concentration':conc,'concentration_unit':concunit,'endpoint_label':el})
        return out
        
    def _endpoints(self, chems, species):
        # return all effect data, (chemical, species, death rate)
        out = []
        for c in chems:
            for s in species:
                l_ = self._endpoint(c,s)
                for e,cc,cu,el in l_:
                    d = {'cas':c, 'taxon':s, 'endpoint':e, 'concentration':cc, 'concentration_unit':cu,'endpoint_label':el}
                    out.append(d)
        return out
    
    def _save(self, directory = './'):
        if self.endpoint:
            raise NotImplementedError ('save from endpoint is not implemented.')
        else: 
            self.graph.serialize(directory + 'ecotox.' + FORMAT , format = FORMAT)
     
    def _get_same_as_species(self, i):
        q = prefixes(self.initNs)
        q += """
            SELECT ?s WHERE {
                <%s> owl:sameAs ?s ;
                     rdf:type nso:Taxon .
            }
        """
        return self._query(q, ['s'])
    
    def _get_same_as_chemical(self, i):
        q = prefixes(self.initNs)
        q += """
            SELECT ?s WHERE {
                <%s> owl:sameAs ?s ;
                     rdf:type nso:Chemical .
            }
        """
        return self._query(q, ['s'])
    
    def _map_ncbi_taxid(self, taxids):
        tmp = get_wikidata_name(taxids)
        out = {}
        for i,s in tmp:
            q = prefixes(self.initNs)
            q += """
                SELECT DISTINCT ?id WHERE {
                ?id rdfs:label ?name ;
                    rdf:type nso:Taxon .
                FILTER regex(?name, "%s", "i")
                }
            """ % str(s)
            
            out[i] = self._query(q, ['id'])
        return out
    
    def _chemical_to_species(self, c):
        q = prefixes(self.initNs)
        q += """
            SELECT ?s {
            VALUES ?c { <%s> }
                ?c nso:effects [
                        nso:species ?s 
                        ] .
            }
        """ % c
        results = self._query(q, ['s'])
            
        return set(results)
    
    def _specie_to_chemicals(self, s):
        q = prefixes(self.initNs)
        q += """
            SELECT ?c {
                VALUES ?s { <%s> }
                ?c nso:effects [
                        nso:species ?s 
                        ] .
            }
        """ % s
        results = self._query(q, ['c'])
            
        return set(results)
    
    def _chemicals_to_species(self, chems):
        return {c:self._chemical_to_species(c) for c in chems}
    
    def _species_to_chemicals(self, species):
        return {c:self._specie_to_chemicals(c) for c in species}
    
    def _classes(self):
        q = prefixes(self.initNs)
        q += """
            SELECT ?c WHERE {
                ?c a owl:Class .
            }
        """
        results = self._query(q, ['c'])
        return set(results)
    
    def _class_label(self, c):
        q = prefixes(self.initNs)
        q += """
            SELECT ?l WHERE {
                <%s> rdfs:label ?l .
            } 
        """ % c
        return self._query(q, ['l'])
    
    ### PUBLIC FUNCTIONS
    ### convert - add namespace/remove namespace
    
    def unit_convertion(self, value, unit):
        """
        Converts a value into standard units.
        args:
            value :: float
            unit :: str 
        return
            float, value in new unit.
        """
        if unit in self.units:
            factor = self.units[unit]
        else:
            tmp = [(k,len(set(str(unit)) & set(str(k)))/len(set(str(unit)) | set(str(k)))) for k in self.units]
            tmp = sorted(tmp, key=lambda x:x[1], reverse = True)
            k,s = tmp.pop(0)
            factor = self.units[k]
            
        return value*factor
    
    def add_same_as_chemicals(self, ids, convert = False):
        """
        Adds sameAs relations between all chemicals in ids.
        args:
            ids :: list
            convert :: boolean
                whether to add namespace to ids
        """
        if convert: ids = [self.namespace['chemical/'+str(s)] for s in ids]
        for a,b in itertools.product(ids, ids):
            self.graph.add((URIRef(a), OWL.sameAs, URIRef(b)))
            
    def add_same_as_species(self, ids, convert = False):
        """
        Adds sameAs relations between all species in ids.
        args:
            ids :: list
            convert :: boolean
                whether to add namespace to ids
        """
        if convert: ids = [self.namespace['taxon/'+str(s)] for s in ids]
        for a,b in itertools.product(ids, ids):
            self.graph.add((URIRef(a), OWL.sameAs, URIRef(b)))
            
    def classes(self):
        """
        All classes of species.
        returns:
            str 
                class URI
        """
        return self._classes()
    
    def class_label(self, c):
        """
        Returns name of class.
        args:
            c :: str 
                class URI
        returns:
            str
        """
        return self._class_label(c)
    
    def hierarchy(self, ids, convert = False):
        """
        Construct hierarchy of species.
        args:
            ids :: list 
                list of species
            convert :: boolean
                whether to add namespace to ids
        returns:
            out :: list
                list of child, parent, score triples stored in dict.
        """
        if convert: ids = [self.namespace['taxon#'+str(s)] for s in ids]
        out = self._hierarchy(ids)
        if convert: out = [{k:strip(d[k], ['#', '/']) for k in d} for d in out]
        return out
    
    def construct_tree(self, subset = None, convert = False):
        """
        Synonym for hierarchy.
        """
        return self.hierarchy(subset, convert)
    
    def map_ncbi_taxid(self, taxids, convert = False):
        """
        Convert ecotox tax ids to ncbi tax ids
        args:
            taxids :: list
                list of ecotox tax ids
            convert :: boolean
                whether to remove namespace from ids
        returns:
            out :: dict
                key - ecotox taxid
                item - ncbi taxid(s)
        """
        if convert: taxids = [t.split('#')[-1] for t in taxids]
        out = self._map_ncbi_taxid(taxids)
        if convert: out = {k:[strip(o,'#') for o in out[k]] for k in out}
        return out
        
    def get_same_as_species(self, ids, convert = False):
        """
        Get identifiers of all equal species.
        args:
            ids :: list
                species
            convert :: boolean
                whether to add namespace to input.
        returns:
            out :: dict
                key - species
                item - list of species that are the same as key.
        """
        if convert: ids = [self.namespace['taxon/'+str(s)] for s in ids]
        try:
            out = {i:set(self._get_same_as_species(i)) for i in ids}
        except TypeError:
            return set()
        if convert: out = {strip(k,['/','#']):[strip(a,['/','#']) for a in out[k]] for k in out}
        return out
     
    def get_same_as_chemical(self, ids, convert = False):
        """
        Get identifiers of all equal chemicals.
        args:
            ids :: list
                chemicals
            convert :: boolean
                whether to add namespace to input.
        returns:
            out :: dict
                key - chemical
                item - list of chemicals that are the same as key.
        """
        if convert: ids = [self.namespace['chemical/'+str(s)] for s in ids]
        try:
            out = {i:set(self._get_same_as_chemical(i)) for i in ids}
        except TypeError:
            return set()
        if convert: out = {strip(k,['/','#']):[strip(a,['/','#']) for a in out[k]] for k in out}
        return out
        
    def endpoints(self, chems, species, convert = False):
        """
        Return all experiment endpoint in ecotox involving chems and species.
        args:
            chems :: list
                chemical list
            species :: list
                species list
            convert :: boolean
                whether to add namespace to inputs.
        returns:
            out :: list
                dict containing
                cas, species, endpoint, concentration, concentration_unit
        """
        if chems == None and species == None:
            try:
                out = self._all_endpoints()
                if convert:
                    out = [{k:strip(s[k],['/','#']) for k in s} for s in out]
                return out
            except AttributeError:
                if chems == None:
                    chems = self.chemicals(convert = convert)
                if species == None:
                    species = self.species(convert = convert)
        if convert:
            chems = [self.namespace['chemical/'+str(c)] for c in chems]
            species = [self.namespace['taxon/'+str(s)] for s in species]
        out = self._endpoints(chems, species)
        if convert:
            out = [{k:strip(s[k],['/','#']) for k in s} for s in out]
        return out
        
    def species(self, convert = False):
        """
        Return all species in dataset.
        args:
            convert :: boolean
                whether to strip namespace on output.
        returns:
            out :: set
                all species
        """
        out = self._species()
        if convert: out = set([strip(s,['/','#']) for s in out])
        return out
        
    def chemicals(self, convert = False):
        """
        Return all chemicals in dataset.
        args:
            convert :: boolean
                whether to strip namespace on output.
        returns:
            out :: set
                all chemicals
        """
        out = self._chemicals()
        if convert: out = set([strip(s,['/','#']) for s in out])
        return out
    
    def chemicals_to_species(self, chems, convert = False):
        """
        Return all species a chemical has an effect on.
        args:
            chems :: list
                chemicals
            convert :: boolean 
                whether to add namespace to input.
        returns:
            out : dict
                key - chemical
                item - effected species
        """
        if convert: chems = [self.namespace['chemical/'+str(c)] for c in chems]
        out = self._chemicals_to_species(chems)
        out = {k:out[k] for k in out if out[k]}
        if convert: out = {strip(str(k),['/','#']):[strip(o,['/','#']) for o in out[k]] for k in out}
        return out
    
    def species_to_chemicals(self, species, convert = False):
        """
        Return all chemicals that has an effect of a species.
        args:
            species :: list
                chemicals
            convert :: boolean 
                whether to add namespace to input.
        returns:
            out : dict
                key - species
                item - effected by chemical
        """
        if convert: species = [self.namespace['taxon/'+str(s)] for s in species]
        out = self._species_to_chemicals(species)
        if convert: out = {strip(k,['/','#']):[strip(o,['/','#']) for o in out[k]] for k in out}
        return out
    
    def species_names(self, subset = None, convert = False):
        """
        Return names of species.
        args:
            subset :: list
                species
            convert :: boolean
                whether to add namespace to input
        returns:
            out :: dict
                key - species
                item - list of names
        """
        subset = set(subset)
        if convert: subset = set([self.namespace['taxon/'+str(s)] for s in subset])
        subset |= self.get_same_as_species(subset)
        out = self._species_names(subset)
        if convert: out = {strip(k,['/','#']):[strip(o,['/','#']) for o in out[k]] for k in out}
        return out 
        
    def chemical_names(self, subset = None, convert = False):
        """
        Return names of chemicals.
        args:
            subset :: list
                chemicals
            convert :: boolean
                whether to add namespace to input
        returns:
            out :: dict
                key - chemical
                item - list of names
        """
        subset = set(subset)
        if convert: subset = set([self.namespace['chemical/'+str(s)] for s in subset])
        subset |= self.get_same_as_chemical(subset)
        out = self._chemical_names(subset)
        if convert: out = {strip(k,['/','#']):[strip(o,['/','#']) for o in out[k]] for k in out}
        return out 
        
    def save(self, directory = None):
        """
        Save loaded data to files.
        args:
            directory :: str
        """
        return self._save(directory)
    
    def query(self, q, var):
        """
        Run any query over the db.
        args:
            q :: str 
                query
            var :: list
                list of variables to return from query.
        returns:
            list 
                list of tuples of len(var)
        
        eg.
        >>> q = 'select ?s ?c where {?s a ?c}'
        >>> var = ['s','c']
        >>> query(q, var)
        [(123, animal), (321, fish)]
        """
        q = prefixes(self.initNs) + q
        return self._query(q, var)
        

        

