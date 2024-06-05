import json
import os

import yaml
from yaml import BaseLoader

dirs = os.listdir('data/office31/amazon')
f = open('class_map.json', 'w')
f.write('{\n')
for i, dir in enumerate(dirs):
    f.write('\t\"'+dir+'\"'+':'+' '+str(i)+',\n')
    f.write('\t\"'+str(i)+'\"'+':'+' '+'\"'+dir+'\"'+',\n')
f.write('}')