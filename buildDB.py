#! python3

import psycopg2
import json
import re

conn = psycopg2.connect("dbname=alloy user=ricardo")
cur = conn.cursor()
dir = 'alloy_analyzer/'
with open(dir + 'file_name.json', 'r') as fr:
    file_name = json.load(fr)

with open(dir + 'file_path.json', 'r') as fr:
    file_path = json.load(fr)

with open(dir + 'line/file_hash.json', 'r') as fr:
    file_hash = json.load(fr)

file_prefix = {}

for docid,name in file_name.items():
    name = name.split('/')[-1]
    path = file_path[docid]
    prefix = re.split('[ .]', name)[0]
    file_prefix[docid] = prefix
    hash_value = file_hash[docid]
    print(docid)
    print(hash_value)
    cur.execute('INSERT INTO als(docid, file_name, path, prefix, hash_value) VALUES(%s, %s, %s, %s, %s);', (docid, name, path, prefix, hash_value))


conn.commit()

cur.close()
conn.close()