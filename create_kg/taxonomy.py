
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import FOAF, RDF, OWL, RDFS
from rdflib.plugins.sparql import prepareQuery
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE
from .utils import prefixes, read_data, test_endpoint, query_endpoint, query_graph, length_file
import warnings
from .utils import strip as strip_
from fuzzywuzzy import fuzz
from tqdm import tqdm

import itertools

FORMAT = 'nt'

def wikidata_common_name(i):
    endpoint = 'https://query.wikidata.org/sparql'
    q = """
    SELECT ?name WHERE {
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    ?item wdt:P685 "%s" . 
    ?item wdt:P225 ?name .
    }
    """ % str(i)
    try:
        res = query_endpoint(endpoint, q, var = ['name'])
    except:
        return []
    
    out = []
    for name in res:
        out.append(name)
    return out

replace_mapping = {}

def strip_compair(name,n):
    # add expert rules, eg. sp.
    for k in replace_mapping:
        if k in name:
            pass 
        if k in n:
            pass
    return fuzz.ratio(name.lower(), n.lower())/100

class Taxonomy:
    """
    Read and manipulate the NCBI taxonomy.
    """
    def __init__(self, directory = '', endpoint = '', namespace = 'http://example.org/ncbi'):
        """
        args:
            directory :: str 
                optional, path to files
            endpoint :: str 
                optional, host for data endpoint.
            namespace :: str
                default namespace to use for rdf data.
        """
        try:
            assert directory or endpoint
        except AssertionError:
            raise ValueError ('Neither directory or endpoint was provided.')
        finally:
            if directory:
                print('Loading data from directory: ' + directory)
            elif endpoint:
                print('Loading data from endpoint: ' + endpoint)            
        
        if not test_endpoint(endpoint):
            self.endpoint = ''
        
        self.hierarchy_file = directory + 'nodes.dmp'
        self.names_file = directory + 'names.dmp'
        self.division_file = directory + 'division.dmp'
        self.ssd_file = directory + 'ssd.txt'
        
        self.namespace = Namespace(namespace+'/')
        self.onto_namespace = Namespace(namespace+'#')
        self.endpoint = endpoint
        self.initNs = {'rdf':RDF, 'foaf':FOAF, 'ns': self.namespace, 'nso':self.onto_namespace, 'owl':OWL, 'rdfs':RDFS}
        
        if not self.endpoint:
            self.graph = Graph()
            self._load_hierarchy()
            self._load_names()
            self._load_division()
            #self._simple_entailment()
            #self._load_ssd()
        
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
        
    ### DATA LOADING FUNCTIONS
        
    def _simple_entailment(self):
        # sameAs
        for s, o in self.graph.subject_objects(predicate=OWL.sameAs):
            self.graph.add((o, OWL.sameAs, s))
        for s, o in self.graph.subject_objects(predicate=OWL.sameAs):
            for s1, p1 in self.graph.subject_predicates(object = s):
                self.graph.add((s1,p1,o))
            for p2, o2 in self.graph.predicate_objects(subject = s):
                self.graph.add((o,p2,o2))
        
    def _load_hierarchy(self):
        # columns in nodes.dmp file of Taxonomy. Only use the first three columns (currently).
        data = read_data(self.hierarchy_file)
        with tqdm(total=length_file(self.hierarchy_file), desc='Loading hierarchy') as pbar:
            for d in data:
                # id, parent, rank, division
                i,p,r,di = d[0], d[1], d[2], d[4]
                i = self.namespace['taxon/'+str(i)]
                self.graph.add((i, RDF.type, self.onto_namespace['Taxon']))
                if r:
                    rt = r.replace(' ','_')
                    self.graph.add((i, self.onto_namespace['rank'], self.namespace['rank/'+rt]))
                    self.graph.add((self.namespace['rank/'+rt], RDFS.label, Literal(r)))
                    self.graph.add((self.namespace['rank/'+rt], RDF.type, self.onto_namespace['Rank']))
                if p:
                    p = self.namespace['taxon/'+str(p)]
                    self.graph.add((i, RDFS.subClassOf, p))
                if di:
                    di = self.namespace['division/'+str(di)]
                    self.graph.add((i, RDFS.subClassOf, di))
                    
                pbar.update(1)
                
    
    def _load_names(self):
        data = read_data(self.names_file)
        with tqdm(total=length_file(self.names_file), desc='Loading names') as pbar:
            for d in data:
                id_, name, unique_name, type_ = d[0], d[1], d[2], d[3]
                id_ = self.namespace['taxon/'+str(id_)]
                name = Literal(name)
                unique_name = Literal(unique_name)
                type_label = Literal(type_)
                type_ = self.onto_namespace[type_.replace(' ','_')]
                
                if unique_name:
                    self.graph.add( (id_, RDFS.label, unique_name) )
                if name:
                    self.graph.add( (id_, type_, name) )
                    self.graph.add( (type_, RDFS.label, type_label) )
                
                pbar.update(1)
                    
                
    
    def _load_division(self):
        data = read_data(self.division_file)
        all_divisions = set()
        for d in data:
            # id, acronym, full
            id_,k,i = d[0], d[1], d[2]
            id_ = URIRef(self.namespace['division/'+ str(id_)])
            self.graph.add((id_, RDFS.label, Literal(i)))
            self.graph.add((id_, RDF.type, self.onto_namespace['Division']))
            self.graph.add((id_, RDF.type, OWL.Class))
                
            all_divisions.add(id_)
            
    def _load_ssd(self):
        data = read_data(self.ssd_file)
        for d in data:
            # id, ssd group
            i, s = d[0], d[1]
            i = self.namespace['taxon/'+str(i)]
            s = self.namespace['SSD/'+str(i)]
            self.graph.add((i, RDF.type, s))
            self.graph.add((s, RDF.type, OWL.Class))
            self.graph.add((s, RDF.type, self.onto_namespace['SSDClass']))
        
    ### PRIVATE FUNCTIONS
    def _query(self, q, var):
        if self.endpoint:
            results = query_endpoint(self.endpoint, q, var = var)
        else:
            results = query_graph(self.graph, q)
        
        if len(var) < 2:
            results = [r[0] for r in results]
        
        return results
    
    def _species(self):
        # return all species
        q = prefixes(self.initNs)
        q += """
            SELECT DISTINCT ?c WHERE {
                ?c rdf:type nso:Species .
            }
        """
        self._query(q, var = ['c'])          
        if len(results) == 10000:
            warnings.warn('Only 10k species will be loaded from endpoint.')
            
        out = set()
        for row in results:
            out.add(row)
        return out
    
    def _names(self, subset = None):
        # return names of species.
        q = prefixes(self.initNs)
        if subset:
            q += """
                SELECT ?c ?n ?t ?un {
                    values ?c { %s }
                    ?c rdf:type nso:Taxon ;
                    OPTIONAL { ?c rdfs:label ?un }
                    OPTIONAL { ?c ?nt ?n .
                               ?nt rdfs:label ?t ;
                                   rdfs:subPropertyOf rdfs:label .
                    }
                }
            """ % '{0}'.format(' '.join(['<'+str(s)+'>' for s in subset]))
        else:
            q += """
            SELECT ?c ?n ?t ?un WHERE {
                ?c rdf:type nso:Taxon ;
                    OPTIONAL { ?c rdfs:label ?un }
                    OPTIONAL { ?c ?nt ?n .
                               ?nt rdfs:label ?t ;
                                   rdfs:subPropertyOf rdfs:label .
                    }
            }
        """
        results = self._query(q, var = ['c', 'n', 't', 'un'])
        out = defaultdict(list)
        
        for s,n,t,un in results:
            out[s].append({t:n})
                
        for s in out:
            d = defaultdict(list)
            for dict_ in out[s]:
                for k,i in dict_.items():
                    d[k].append(i)
            out[s] = d
        
        return out
    
    def _construct_tree(self, subset = None):
        # return set of (child,parent) pairs.
        q = prefixes(self.initNs)
        
        sparql = SPARQLWrapper(self.endpoint)
        if subset:
            q += """
                SELECT ?c ?p {
                    values ?c { %s }
                     ?c rdfs:subClassOf ?p ;
                        a nso:Taxon .
                }
            """ % '{0}'.format(' '.join(['<'+str(s)+'>' for s in subset]))
        else:
            q += """
                SELECT ?c ?p WHERE {
                    ?c rdfs:subClassOf ?p ;
                        a nso:Taxon .
                }
            """
        
        result = self._query(q, var = ['c','p'])
        out = []
        for c,p in result:
            out.append({'child':c, 'parent':p, 'score':1})
        
        return out
    
    def _sibling(self, cid):
        q = prefixes(self.initNs)
        q +=  """
            SELECT ?p {
                VALUES ?s {<%s>}
                    {   ?s rdfs:subClassOf [
                        ^rdfs:subClassOf ?p
                        ]   .  
                    }
                    UNION 
                    {   ?s rdf:type [
                        ^rdf:type ?p
                        ]   . 
                    }
                }
            """ % cid
        
        result = self._query(q, var = ['p'])
        
        out = set()
        for row in result:
            out.add(row)
        return out
    
    def _siblings(self, species):
        return {s:self._sibling(s) for s in species}
    
    def _traverse_help(self, l, sb, species, down):
        out = set()
        s = ''
        for k in range(l + 1):
            if k == l:
                s += sb
            else:
                s += sb+'/'
        q = prefixes(self.initNs)
        a = '{0}'.format(' '.join(['<'+str(s)+'>' for s in species]))
        if down:
            q += """SELECT DISTINCT ?s {
                values ?x { %s }
                { ?s %s ?x }
                }"""  % (a,s)
        else:
            q += """SELECT DISTINCT ?s {
                values ?x { %s }
                { ?x %s ?s }
                }"""  % (a,s)
        
        
        result = self._query(q, var = ['s'])
        
        for r in result:
            out.add(r)
        return out
    
    def _traverse(self, species, levels = 1, down = False):
        if not species:
            return self._species()
        
        out = set()
        a = ''
        if down:
            a = '^'
        
        initNs = {'ns':self.namespace}
        q = prefixes(initNs)
        
        out = set()
        l = 0
        
        length = len(species)
        
        if levels == -1: # to the bottom/top
            return self._traverse(species, levels = float('inf'), down = down)
            
        else: 
            while l < levels:
                out |= self._traverse_help(l, 'rdfs:subClassOf', species, down)
                if len(out) <= length:
                    break
                length = len(out)
                l += 1
            return out
    
    def _subset(self, species, up_levels = 1, down_levels = 1, convert = False, strip = False):
        if convert: species = [self.namespace['id#'+str(s)] for s in species]
        out = self._traverse(species, up_levels, down = False) | self._traverse(species, down_levels, down = True) | set(species)
        if strip: out = set([strip(s, '#') for s in out])
        return out
    
    def _save(self, directory):
        if self.endpoint:
            raise NotImplementedError ('save from endpoint is not implemented.')
        else:
            self.graph.serialize(directory + 'taxonomy.' + FORMAT, format = FORMAT)
            
    def _classes(self):
        q = prefixes(self.initNs)
        q += """
            SELECT ?c ?n ?l WHERE {
                ?c a owl:Class .
                OPTIONAL {?c rdfs:label ?l}
            }
        """ 
        res = self._query(q, var = ['c','n','l'])
        return res 
    
    def _ssds(self):
        q = prefixes(self.initNs)
        q += """
            SELECT ?c ?n ?l WHERE {
                ?c a owl:Class ;
                    a nso:SSDClass .
                OPTIONAL {?c rdfs:label ?l}
            }
            ORDER BY ?c
        """
        return self._query(q, var = ['c','n','l'])
        
    def _divisions(self):
        q = prefixes(self.initNs)
        q += """
            SELECT ?c ?n ?l WHERE {
                ?c a owl:Class ;
                    a nso:Division .
                OPTIONAL {?c rdfs:label ?l}
            }
            ORDER BY ?c
        """
        return self._query(q, var = ['c','n','l'])
    
    
    def _class_single(self, class_uri):
        out = []
        res = []
        i = 1
        while res or i == 1:
            q = prefixes(self.initNs)
            
            if i == 1:
                s1 = 'rdfs:subClassOf'
                s2 = 'rdf:type'
            else:
                s1 = '/'.join(['rdfs:subClassOf'] * i)
                s2 = '/'.join(['rdf:type'] * i)
            
            q += """
            SELECT ?s {
                VALUES ?c {<%s>}
                ?s rdf:type nso:Species .
                    { ?s %s ?c . }
                    UNION 
                    { ?s %s ?c . }
                }
            """ % (str(class_uri), s1, s2)
            res = self._query(q, var = ['s'])
            out.extend(res)
            i += 1
        return out
    
    def _class_(self, class_ids):
        return {c:self._class_single(c) for c in class_ids}
    
    def _levels(self):
        q = prefixes(self.initNs)
        q += """
            SELECT DISTINCT ?r ?l WHERE {
                ?r rdf:type nso:Rank 
                OPTIONAL { ?r rdfs:label ?l}
            }
        """
        out = self._query(q, ['r', 'l'])
        return out
        
    def _ids(self, name, rank):
        q = prefixes(self.initNs)
        levels = [a for a,_ in self._levels()]
        use_rank = any([str(rank) == str(a) for a in levels])
        rank = str(rank)
        if use_rank:
            q += """
                SELECT ?s ?n WHERE {
                    ?s rdf:type nso:Taxon ;
                            rdf:type <%s> ;
                            rdfs:label ?n .
                        OPTIONAL {?s ?nt ?n .
                                ?nt rdfs:subPropertyOf rdfs:label .}
                    FILTER(regex(STR(?n), "%s", 'i'))
                }
            """ % (rank, name)
        else:
            q += """
                SELECT ?s ?n WHERE {
                    ?s rdf:type nso:Taxon ;
                            rdf:type <%s> ;
                            rdfs:label ?n .
                        OPTIONAL {?s ?nt ?n .
                                ?nt rdfs:subPropertyOf rdfs:label .}
                    FILTER(regex(STR(?n), "%s", 'i'))
                }
            """ % name
        out = list(self._query(q, ['s','n']))
        return out
    
    def _ids_any(self, name, rank):
        offset = 0
        res = []
        out = []
        levels = [a for a,_ in self._levels()]
        use_rank = any([str(rank) == str(a) for a in levels])
        rank = str(rank)
        while res or offset == 0:
            q = prefixes(self.initNs)
            if use_rank:
                q += """
                    SELECT ?s ?n WHERE {
                        ?s rdf:type nso:Taxon ;
                            rdf:type <%s> ;
                            rdfs:label ?n .
                        OPTIONAL {?s ?nt ?n .
                                ?nt rdfs:subPropertyOf rdfs:label .}
                    } 
                    ORDER BY ?s
                    OFFSET %s
                """ % (rank, str(offset))
            else:
                q += """
                    SELECT ?s ?n WHERE {
                        ?s a nso:Taxon ;
                        OPTIONAL {?s ?nt ?n .
                                ?nt rdfs:subPropertyOf rdfs:label .}
                    }
                    ORDER BY ?s
                    OFFSET %s
                """ % str(offset)
            res = list(self._query(q, ['s','n']))
            out.extend(res)
            offset += 10000
        return out
        
    ### PUBLIC FUNCTIONS
    
    def divisions(self, strip = False):
        """
        Get division domains.
        """
        res = self._divisions()
        out = []
        for c,n,l in res:
            if strip: c = strip_(c,['/','#'])
            out.append({'id':c,'name':n,'label':l})
        return out
    

    def ids(self, name, rank = None, convert = False, strip = False, top_n = -1):
        """
        Get ids of name. Will score full partial matches.
        Args:
            name :: string
            rank :: string
                optional, taxonomic level to search
            convert :: boolean
                add namespace to input rank
            strip :: boolean
                remove namespace from output
        Returns:
            out :: list 
                list of ids that matched
        """
        if convert and rank: rank = self.namespace['rank/'+rank]
        out = self._ids(name, rank = rank)
        out = [{'id':i, 'name':n, 'score':strip_compair(name,n)} for i,n in out]
        if strip: out = [{k:strip_(d[k],['/','#']) for k in d} for d in out]
        out = sorted(out, key = lambda d: d['score'], reverse=True)
        return out[:top_n]
    
    def ids_any(self, name, rank = None, convert = False, strip = False, top_n = -1):
        """
        Get ids of name. Will score any match.
        Args:
            name :: string
            rank :: string
                optional, taxonomic level to search
            convert :: boolean
                add namespace to input rank
            strip :: boolean
                remove namespace from output
        Returns:
            out :: list 
                list of ids that matched
        """
        out = self.ids(name = name, rank = rank, convert = convert, strip = strip, top_n = top_n)
        if out:
            return out
        if convert and rank: rank = self.namespace['rank/'+rank]
        out = self._ids_any(name, rank = rank)
        out = [{'id':i, 'name':n, 'score':strip_compair(name,n)} for i,n in out]
        if strip: out = [{k:strip_(d[k],['/','#']) for k in d} for d in out]
        out = sorted(out, key = lambda d: d['score'], reverse=True)
        return out[:top_n]
    
    
    def levels(self, strip = False):
        """
        Return levels names and id.
        Args:
            strip :: boolean
                strip namespace of output
        Returns:
            out :: list
                list of levels
        """
        res = self._levels()
        out = []
        for r,l in res:
            if strip: r = strip_(r, ['/','#'])
            out.append({'id':r, 'label':l})
        return out
    
    def classes(self, strip = False):
        """
        Returns all classes in dataset.
        """
        out = []
        res = self._classes()
        for c,n,l in res:
            if strip: c = strip_(c,['/','#'])
            out.append({'id':c,'name':n,'label':l})
        return out
    
    def class_name(self, c):
        """
        Return class name.
        Args: 
            c :: str
                class URI
        """
        return self._class_name(c)
    
    def ssds(self, strip = False):
        """
        Returns all ssds.
        """
        out = []
        res = self._ssds()
        for c,n,l in res:
            if strip: c = strip_(c,['/','#'])
            out.append({'id':c,'name':n,'label':l})
        return out
    
    def save(self, directory = './'):
        """
        Save data to RDF.
        Args:
            directory :: str 
                path to save files at.
        """
        self._save(directory)
        
    def names(self, subset = None, convert = False, strip = False):
        """
        Get names and synonyms for species.
        Args:
            subset :: set
                set of species 
            convert :: boolean
                whether to add namespace to subset. 
            strip :: boolean
                whether to strip namespace on output.
        Returns:
            out :: dict 
                key - species id
                item - list of names
        """
        if convert: subset = [self.namespace['taxon/'+str(s)] for s in subset]
        out = self._names(subset)
        if strip: out = {strip_(k,['/','#']):out[k] for k in out}
        return out
        
    def species(self, strip = False):
        """
        All species.
        Args:
            strip :: boolean
                remove namespace from output
        Returns:
            out :: list
                all species in dataset.
        """
        out = self._species()
        if strip: out = set([strip_(s,['/','#']) for s in out])
        return out
    
    def construct_tree(self, subset = None, convert = False, strip = False):
        """
        Construct hierarchy.
        Args:
            subset :: set
                set of species 
            convert :: boolean
                whether to add namespace to subset. 
            strip :: boolean
                whether to strip namespace on output.
        Returns:
            out :: list of dicts
                keys - (child, parent, score)
                items - (id, id, adjacency) 
        """
        if convert: subset = [self.namespace['taxon/'+str(s)] for s in subset]
        out = self._construct_tree(subset)
        if strip: out = [{k:strip_(s[k],['#','/']) for k in s} for s in out]
        return out
    
    def siblings(self, species, convert = False, strip = False):
        """
        Construct hierarchy.
        Args:
            species :: set
                set of species 
            convert :: boolean
                whether to add namespace to subset. 
            strip :: boolean
                whether to strip namespace on output.
        Returns:
            out :: list of dicts
                key - species
                items - list of siblings 
        """
        if convert: species = [self.namespace['taxon/'+str(s)] for s in species]
        out = self._siblings(species)
        if strip: out = {strip_(s, ['/','#']):[strip_(k,['/','#']) for k in out[s]] for s in out}
        return out
    
    def subset(self, species, up_levels = 1, down_levels = 1, convert = False, strip = False):
        """
        Find all species or classes during a traverse of tree.
        Args:
            species :: set
                set of species
            up_levels :: int
                number of levels to traverse upward. -1 denotes to the top.
            down_levels :: int
                number of levels to traverse downward. -1 denotes to the bottom.
            convert :: boolean
                whether to add namespace to subset. 
            strip :: boolean
                whether to strip namespace on output.
        Returns:
            tmp1 :: set
                set of all species passed during traverse.
        """
        if convert: species = [self.namespace['taxon/'+str(s)] for s in species]
        tmp1 = self._subset(species, up_levels, down_levels)
        tmp2 = self.siblings(tmp1)
        for k in tmp2:
            tmp1 |= tmp2[k]
        if strip: tmp1 = set([strip_(s,['/','#']) for s in tmp1])
        return tmp1
    
    def class_(self, class_ids, convert = False, strip = False):
        """
        Return all species in a class.
        Args:
            class_ids :: list
                ids or URI for class
            convert :: boolean
                whether to add namespace
            strip :: boolean
                whether to strip namespace
        Returns:
            out :: dict
                key - class_id
                item - list of species ids
        """
        
        if convert: class_ids = [self.namespace[str(s)] for s in class_ids]
        out = self._class_(class_ids)
        if strip: out = {strip_(k,['#','/']):[strip_(s,['/','#']) for s in out[k]] for k in out}
        return out
    
    def division(self, class_ids, convert = False, strip = False):
        """
        Return all species in a division.
        Args:
            class_ids :: list
                ids or URI for class
            convert :: boolean
                whether to add namespace
            strip :: boolean
                whether to strip namespace
        Returns:
            out :: dict
                key - class_id
                item - list of species ids
        """
        
        if convert: class_ids = [self.namespace['division/'+str(s)] for s in class_ids]
        out = self._class_(class_ids)
        if strip: out = {strip_(k,['#','/']):[strip_(s,['/','#']) for s in out[k]] for k in out}
        return out
    
    def ssd(self, class_ids, convert = False, strip = False):
        """
        Return all species in a ssd group.
        Args:
            class_ids :: list
                ids or URI for class
            convert :: boolean
                whether to add namespace
            strip :: boolean
                whether to strip namespace
        Returns:
            out :: dict
                key - class_id
                item - list of species ids
        """
        
        if convert: class_ids = [self.namespace['ssd/'+str(s)] for s in class_ids]
        out = self._class_(class_ids)
        if strip: out = {strip_(k,['#','/']):[strip_(s,['/','#']) for s in out[k]] for k in out}
        return out
    
    def query(self, q, var):
        q = prefixes(self.initNs) + q
        return self._query(q,var)
    
    
    
