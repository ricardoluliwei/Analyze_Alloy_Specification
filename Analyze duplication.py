import sys
import math
import tokenizer
from pathlib import Path
from collections import defaultdict
import json
import re
from simhash import Simhash
from itertools import combinations
from sklearn.cluster import KMeans
import numpy as np


def read_data_directory(data_dir_path):
    file_path = {}
    file_name = {}
    
    try:
        with open("file_name.json", "r") as file:
            file_name = json.load(file)
        
        with open("file_path.json", "r") as file:
            file_path = json.load(file)
        
        return file_name, file_path
    
    except:
        data_dir_path = Path(data_dir_path)
        model_dir_paths = []
        if data_dir_path.is_dir():
            for i in data_dir_path.iterdir():
                if i.is_dir():
                    model_dir_paths.append(i)
        
        docid = 0
        
        for d in model_dir_paths:
            for f in d.iterdir():
                if f.is_file():
                    file_path[str(docid)] = str(f.absolute())
                    file_name[str(docid)] = str(f)
                    docid += 1
        
        with open("file_path.json", "w") as file:
            json.dump(file_path, file)
        
        with open("file_name.json", "w") as file:
            json.dump(file_name, file)
    
    return file_name, file_path


def removeComments(text):
    text = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "",
                  text)  # remove comment blocks
    text = re.sub(re.compile("//.*?\n"), "", text)  # remove single line comment
    return text


'''
return a dictionary, key is docid, value is its simhash value
mode: "word" tokenized by word
      "line" tokenized by line
      "block" tokenized by block
'''


def find_file_hashs(file_path, mode: str):
    file_hash = {}
    mode_dir = Path(mode)
    if not mode_dir.exists():
        mode_dir.mkdir()
    
    file_hash_path = mode_dir / "file_hash.json"
    try:
        with open(file_hash_path, "r") as file:
            file_hash = json.load(file)
    
    except:
        for i, j in file_path.items():
            with open(j, "r", encoding="ISO-8859-1") as file:
                text = file.read()
                text = removeComments(text)
                tokens = tokenizer.tokenize(text, mode)
                file_hash[str(i)] = Simhash(tokens).value
        
        with open(file_hash_path, "w") as file:
            json.dump(file_hash, file)
    
    finally:
        return file_hash


'''
create a dictionary, key is hash value of a file, value is a list of docid
'''


def find_exact_duplicate(file_hash, mode: str):
    duplicate = {}
    mode_dir = Path(mode)
    if not mode_dir.exists():
        mode_dir.mkdir()
    
    exact_duplicate_path = mode_dir / "exact_duplicate.json"
    try:
        with open(exact_duplicate_path, "r") as file:
            duplicate = json.load(file)
            return duplicate
    except:
        for k, v in file_hash.items():
            if v in duplicate.keys():
                duplicate[v].append(k)
            else:
                duplicate[v] = [k]
        
        duplicate = {k: v for k, v in sorted(duplicate.items(), key=lambda kv:
        len(kv[1]), reverse=True)}
        
        with open(exact_duplicate_path, "w") as file:
            json.dump(duplicate, file)
        
        return duplicate


'''
make combinations of each pair of file and find similarities score between them
return a dictionary
example:
{
(0,1): 0.33
(0,2): 0.1
...
(111, 123): 0.99
}
'''


def find_similarity(mode: str):
    mode_dir = Path(mode)
    if not mode_dir.exists():
        mode_dir.mkdir()
    
    file_hash_path = mode_dir / "file_hash.json"
    
    with open(file_hash_path, "r") as file:
        file_hash = json.load(file)
    
    N = len(file_hash)  # total number of files
    
    pairs = combinations(range(N), 2)
    
    similarity_path = mode_dir / "similarity.json"
    similarity = {}
    try:
        with open(similarity_path, "r") as file:
            similarity = json.load(file)
        return similarity
    except:
        pass
    
    for pair in pairs:
        i, j = pair
        
        hash = file_hash[str(i)] ^ file_hash[str(j)]
        hash = bin(hash)[2:]
        similarity_score = (hash.count("0") + 64 - len(hash)) / len(hash)
        key = str(pair[0]) + "," + str(pair[1])
        similarity[key] = format(similarity_score, ".3f")
    
    with open(similarity_path, "w") as file:
        json.dump(similarity, file)
    
    return similarity


'''
create a dictionary, key is the first item of the group, value is the group that are similiar within the threshold
'''


def find_near_duplicate(mode, threshold_begin, threshold_end, file_name):
    mode_dir = Path(mode)
    if not mode_dir.exists():
        mode_dir.mkdir()
    
    similarity = find_similarity(mode)
    
    near_duplicate = defaultdict(list)
    near_duplicate_path = mode_dir / f"near_duplicate" \
                                     f"{format(threshold_begin, '.2f')}" \
                                     f"-{format(threshold_end, '.2f')}.json"
    group_path = mode_dir / f"group{format(threshold_begin, '.2f')}-{format(threshold_end, '.2f')}.json"
    
    try:
        with open(near_duplicate_path, "r") as file:
            near_duplicate = json.load(file)
            return near_duplicate
    except FileNotFoundError:
        pass
    
    similar_pairs = [k for k, v in similarity.items() if threshold_begin <
                     float(v) <=
                     threshold_end]
    
    group = {}
    group_num = 0
    
    for pair in similar_pairs:
        pair = pair.split(",")
        i = pair[0]
        j = pair[1]
        if i in group.keys():
            group[j] = group[i]
        elif j in group.keys():
            group[i] = group[j]
        else:
            group_num += 1
            group[i] = group_num
            group[j] = group_num
    
    for k, v in group.items():
        near_duplicate[v].append(file_name[str(k)])
    
    with open(group_path, "w") as file:
        json.dump(group, file)
    
    with open(near_duplicate_path, "w") as file:
        json.dump(near_duplicate, file)
    
    return near_duplicate


def k_means_find_group(mode, k):
    mode_dir = Path(mode)
    if not mode_dir.exists():
        raise FileNotFoundError
    
    data = []
    groups = {}
    
    try:
        with open(str(mode_dir / f"kmeans={k}.json"), "r") as file:
            groups = json.load(file)
        
        return groups
    except:
        pass
    
    with open(str(mode_dir / "file_hash.json"), "r") as file:
        file_hash = json.load(file)
    
    with open("file_name.json", "r") as file:
        file_name = json.load(file)
    
    for _, j in file_hash.items():
        binary = [int(c) for c in bin(j)[2:]]
        while len(binary) != 64:
            binary = [0] + binary
        data.append(binary)
    
    data = np.array(data)
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    

    
    for i, j in enumerate(kmeans.labels_):
        if j in groups.keys():
            groups[int(j)].append(file_name[str(i)])
        else:
            groups[int(j)] = [file_name[str(i)]]
    
    groups = {k: v for k, v in sorted(groups.items(), key=lambda x: x[0])}
    
    with open(str(mode_dir/f"kmeans={k}.json"), "w") as file:
        json.dump(groups, file)
        
    return groups


def find_name_groups(file_name):
    pattern = "[ .0-9]"
    groups = {}
    prefix = {}
    num = 0
    for _, name in file_name.items():
        tokens = re.split(pattern, name)
        if tokens[0] in prefix.keys():
            groups[prefix[tokens[0]]].append(name)
        else:
            prefix[tokens[0]] = str(num)
            groups[str(num)] = [name]
            num += 1
        
    
    with open("name_groups.json", "w") as file:
        json.dump(groups, file)
    
    with open("prefix.json", "w") as file:
        json.dump(prefix, file)
    
    return prefix

if __name__ == '__main__':
    modes = ["word", "line", "block"]
    data_path = "Alloy_models"
    
    file_name, file_path = read_data_directory(data_path)
    
    # for mode in modes:
    #     find_file_hashs(file_path, mode)
    #     print(f"finsh find hash on {mode}!")
    #     find_similarity(mode)
    #     print(f"finished find similarity on {mode}")
    #     threshold = 0.6
    #     step = 0.05
    #     while threshold < 1:
    #         find_near_duplicate(mode, threshold, threshold + step, file_name)
    #         print(f"finished find near_duplicate from {threshold} to "
    #               f"{threshold + step}")
    #         threshold += step
    
    np.set_printoptions(threshold=sys.maxsize)
    
    # prefix = find_name_groups(file_name)
    # for k, v in prefix.items():
    #     print(k)
    
    kmeanns839 =  k_means_find_group(modes[0], 839)
    # kmeans1000 = k_means_find_group(modes[0], 1000)
    # kmeans700 = k_means_find_group(modes[0], 700)
    # 
    # for k,v in kmeans700.items():
    #     print(f"{k}: {v}")
    # 
    # find_near_duplicate("word", 0.9, 0.95, file_name)
    
    
    
    # with open("word/near_duplicate0.90-0.95.json", "r") as file:
    #     n = json.load(file)
    #
    # for k, v in n.items():
    #     print(f"{k}: {v}")

    
    print("Done!")
