# species.py

from fuzzywuzzy import fuzz, process
from tqdm import tqdm

def convert_case(set1, set2):
    try:
        set1 = set(map(str.lower, set1))
        set2 = set(map(str.lower, set2))
    except TypeError:
        print(set1, set2)
    return set1, set2

def equal_words(set1, set2, case_insensitive = False):
    if case_insensitive:
        set1, set2 = convert_case(set1, set2)
    # exact equal words
    a = set1.intersection(set2)
    if len(a) > 0:
        return 100
    else:
        return 0

def fuzzy_words(set1, set2, case_insensitive = False):
    # Partial matches, Levinshtein distance
    if case_insensitive:
        set1, set2 = convert_case(set1, set2)
    s = 0
    for s1 in set1:
        for s2 in set2:
            s = max(s, fuzz.ratio(s1, s2))
    return s
    
def word_in(set1, set2, case_insensitive = False):
    if case_insensitive:
        set1, set2 = convert_case(set1, set2)
  
    s = 0
    for s1 in set1:
        for s2 in set2:
            if s1 in s2:
                s = 100
    return s
    
def species_mapping(eco_mapping, tax_mapping, method = equal_words, prop=0.9, case_insensitive = False):
    """
    Map between ecotox and ncbi taxonomy.
    eco_mapping[int] = set(synonyms)
    tax_mapping[int] = set(synonyms)
    
    return d[ecotox_id] = {ncbi_id, similarity score}
    """
    out = {}
    l = len(list(eco_mapping.keys()))
    # progress bar
    with tqdm(total=l) as pbar:
        for k1, i1 in eco_mapping.items():
            for k2, i2 in tax_mapping.items():
                s = method(i1, i2, case_insensitive = case_insensitive)
                if s < prop*100:
                    continue
                if k1 in out:
                    tmp_s = out[k1]['score']
                else:
                    tmp_s = 0
                out[k1] = {'ncbi_taxon_id':k2, 'score':max(tmp_s,s)}
            pbar.update(1)
    return out


