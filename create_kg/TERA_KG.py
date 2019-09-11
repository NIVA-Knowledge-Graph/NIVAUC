from rappt import Taxonomy
from rappt import Chemistry
from rappt import Ecotox
from rappt import strip

from rdflib import Graph
from rdflib.Namespace import OWL

basepath = './'
t = Taxonomy(directory = basepath+'taxdump/', namespace = 'http://example.org/ncbi')
t.save(basepath+'rdf/')

c = Chemistry(directory = basepath+'pubchem/')
c.save(basepath+'rdf/')

e = Ecotox(directory = basepath+'ecotox_ascii_03_14_2019/', namespace = 'http://example.org/ecotox')
e.save(basepath+'rdf/')

### Mapping CAS to CID
chems = [strip(s,['/','#']) for s in e.chemicals()]
cids = c.convert_ids(from_='cas',to_='cid',chems)

sameas_graph = Graph()
for a,b in zip(chems, cids):
    if a and b:
        a = e.namespace['chemical/'+str(a)]
        b = c.compound_namespace['CID'+str(s)]
        sameas_graph.add((a,OWL.sameAs,b))
sameas_graph.serialize(basepath+'rdf/equiv.nt',format='nt')
