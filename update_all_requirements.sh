#!/bin/bash

# updating all dependencies in requirements.txt
# cut -f 1 -d= requirements.txt | paste -d' ' -s - | xargs pip install -U
# create pip freeze output
pip freeze > new.txt
# updating requirements.txt with new versions
awk -F'==' '{print $1}' requirements.txt | xargs -n1 -I {} sh -c "less new.txt | grep {}" | sort | uniq > nrequirements.txt
# remove the temporary freeze file
rm new.txt
rm requirements.txt
mv nrequirements.txt requirements.txt
# check if all dependencies are installed
pip install -r requirements.txt