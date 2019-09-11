## Setup

Install [rappt](https://gitlab.com/Erik-BM/rappt) and follow the instructions on how to download the data.

## Creating the knowledge graph

The script [TERA_KG.py](https://github.com/Erik-BM/NIVAUC/tree/master/create_kg/TERA_KG.py) will create the individual components of the knowledge graph (_i.e._ effect, taxonomy, chemical hierarchy, chemical mapping), these can be concatenated using _e.g._  __cat__ on unix. The mapping between species in ECOTOX and NCBI is created using [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher) and can be found [here](https://github.com/Erik-BM/NIVAUC/tree/master/kg/logmap2_mappings.owl.gz) . 
