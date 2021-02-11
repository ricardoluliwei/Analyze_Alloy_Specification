import sys
import os
from pathlib import Path
import re
import json

directory = Path("Alloy_models/Android")

ignore = ["androidDeclaration", "appDeclaration", ".kk.java", ".cnf"]

pat = re.compile('|'.join(ignore))
failList = []

for f in directory.iterdir():
    if re.search(pat, str(f)):
        continue
    
    status = os.system(f"java -jar smelter.jar \"{str(f)}\" 1")
    
    if status != 0:
        failList.append(str(f))
    
with open("failList.json", "w") as file:
    json.dump(failList, file)
    
