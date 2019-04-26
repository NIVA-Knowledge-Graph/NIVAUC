# Scripts for creating the TERA knowledge graph

This work uses _python3.6_ with requirements that are found [here](https://github.com/Erik-BM/NIVAUC/tree/master/create_kg/requirements.txt) . 

## Downloading raw data
Downloading the raw data from [ECOTOX](https://cfpub.epa.gov/ecotox/), [NCBI](https://www.ncbi.nlm.nih.gov/taxonomy) and [PubChem](https://pubchem.ncbi.nlm.nih.gov/) can be done by excecuting the [download_raw_data.sh](https://github.com/Erik-BM/NIVAUC/tree/master/create_kg/download_raw_data.sh) shell script. 

## Creating the knowledge graph

The script [TERA_KG.py](https://github.com/Erik-BM/NIVAUC/tree/master/create_kg/TERA_KG.py) will create the individual components of the knowledge graph (_i.e._ effect, taxonomy, chemical hierarchy, chemical mapping), these can be concatenated using _e.g._  __cat__ on unix. The mapping between species in ECOTOX and NCBI is created using [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher) and can be found [here](https://github.com/Erik-BM/NIVAUC/tree/master/kg/logmap2_mappings.owl.gz) . 
