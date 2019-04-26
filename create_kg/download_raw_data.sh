
mkdir tmp

wget ftp://newftp.epa.gov/ecotox/ecotox_ascii_03_14_2019.exe
unzip ecotox_ascii_03_14_2019.exe
rm ecotox_ascii_03_14_2019.exe

#enter input encoding here
FROM_ENCODING="Windows-1252"
#output encoding(UTF-8)
TO_ENCODING="UTF-8"
#convert
CONVERT=" iconv  -f   $FROM_ENCODING  -t   $TO_ENCODING"
#loop to convert multiple files 
for  file  in ./ecotox_ascii_03_14_2019/*.txt; do
     $CONVERT   "$file"   -o  "${file%.txt}.utf8.converted"
done
for  file  in ./ecotox_ascii_03_14_2019/validation/*.txt; do
     $CONVERT   "$file"   -o  "${file%.txt}.utf8.converted"
done

wget ftp://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.zip --directory-prefix=tmp/
unzip tmp/new_taxdump.zip -d ./taxdump
rm -r tmp

mkdir pubchem
wget ftp://ftp.ncbi.nlm.nih.gov/pubchem/RDF/compound/general/pc_compound2parent.ttl.gz --directory-prefix=pubchem/
wget ftp://ftp.ncbi.nlm.nih.gov/pubchem/RDF/compound/general/pc_compound_type.ttl.gz --directory-prefix=pubchem/
gzip -d pubchem/*
