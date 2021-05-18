import sys
import math
import tokenizer
import os
from pathlib import Path
from collections import defaultdict
import scipy.special
import json
import re
import matplotlib.pyplot as plt
from simhash import Simhash
from itertools import combinations
from sklearn.cluster import KMeans
import numpy as np


def removeComments(text):
    text = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "",
                  text)  # remove comment blocks
    text = re.sub(re.compile("//.*?\n"), "", text)  # remove single line comment
    return text

class clone_analyzer:
    def __init__(self, name, data_dir_path):
        self.name = name + '/'
        self.data_dir_path = data_dir_path
        if not os.path.exists(name):
            os.makedirs(name)
            
        self.file_name, self.file_path = self.read_data_directory()

    def read_data_directory(self) -> (dict, dict):
        print('Reading data dirctory!')
        file_path = {}
        file_name = {}
        
        path_for_file_path = self.name + "file_path.json"
        path_for_file_name = self.name + 'file_name.json'
        try:
            with open(path_for_file_name, "r") as file:
                file_name = json.load(file)
            
            with open(path_for_file_path, "r") as file:
                file_path = json.load(file)
            print('Done!')
            return file_name, file_path
        
        except:
            data_dir_path = Path(self.data_dir_path)
            
            docid = 0
            if data_dir_path.is_dir():
                for f in data_dir_path.iterdir():
                    if f.is_file():
                        file_path[str(docid)] = str(f.absolute())
                        file_name[str(docid)] = str(f)
                        docid += 1
            
            with open(path_for_file_path, "w") as file:
                json.dump(file_path, file)
            
            with open(path_for_file_name, "w") as file:
                json.dump(file_name, file)
        print('Done!')
        return file_name, file_path


    '''
    return a dictionary, key is docid, value is its simhash value
    mode: "word" tokenized by word
          "line" tokenized by line
          "block" tokenized by block
    '''


    def find_file_hashs(self, mode: str):
        print('Finding hash value for files!')
        file_hash = {}
        mode_dir = Path(self.name + mode)
        if not mode_dir.exists():
            mode_dir.mkdir()
        
        file_hash_path = mode_dir / "file_hash.json"
        try:
            with open(file_hash_path, "r") as file:
                file_hash = json.load(file)
        
        except:
            for i, j in self.file_path.items():
                with open(j, "r", encoding="ISO-8859-1") as file:
                    text = file.read()
                    text = removeComments(text)
                    tokens = tokenizer.tokenize(text, mode)
                    file_hash[str(i)] = Simhash(tokens).value
            
            with open(file_hash_path, "w") as file:
                json.dump(file_hash, file)
        
        finally:
            print('Done!')
            return file_hash

    def get_prefix(self):
        print('Finding prefix of files!')
        dir = Path(self.name)
        prefix_path = dir + 'prefix.json'
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


    def find_similarity(self, mode: str):
        print('Finding similarity score!')
        mode_dir = Path(self.name + mode)
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
            print('Done!')
            return similarity
        except:
            pass
        
        for pair in pairs:
            i, j = pair
            
            hash = file_hash[str(i)] ^ file_hash[str(j)]
            hash = bin(hash)[2:]
            similarity_score = hash.count("0") / len(hash)
            key = str(pair[0]) + "," + str(pair[1])
            similarity[key] = format(similarity_score, ".3f")
        
        with open(similarity_path, "w") as file:
            json.dump(similarity, file)
        print('Done!')
        return similarity


# '''
# create a dictionary, key is the first item of the group, value is the group that are similiar within the threshold
# '''
#
#
# def find_near_duplicate(mode, threshold_begin, threshold_end, file_name):
#     mode_dir = Path(mode)
#     if not mode_dir.exists():
#         mode_dir.mkdir()
#
#     similarity = find_similarity(mode)
#
#     near_duplicate = defaultdict(list)
#     near_duplicate_path = mode_dir / f"near_duplicate" \
#                                      f"{format(threshold_begin, '.2f')}" \
#                                      f"-{format(threshold_end, '.2f')}.json"
#     group_path = mode_dir / f"group{format(threshold_begin, '.2f')}-{format(threshold_end, '.2f')}.json"
#
#     try:
#         with open(near_duplicate_path, "r") as file:
#             near_duplicate = json.load(file)
#             return near_duplicate
#     except FileNotFoundError:
#         pass
#
#     similar_pairs = [k for k, v in similarity.items() if threshold_begin <
#                      float(v) <=
#                      threshold_end]
#
#     group = {}
#     group_num = 0
#
#     for pair in similar_pairs:
#         pair = pair.split(",")
#         i = pair[0]
#         j = pair[1]
#         if i in group.keys():
#             group[j] = group[i]
#         elif j in group.keys():
#             group[i] = group[j]
#         else:
#             group_num += 1
#             group[i] = group_num
#             group[j] = group_num
#
#     for k, v in group.items():
#         near_duplicate[v].append(file_name[str(k)])
#
#     with open(group_path, "w") as file:
#         json.dump(group, file)
#
#     with open(near_duplicate_path, "w") as file:
#         json.dump(near_duplicate, file)
#
#     return near_duplicate


# def k_means_find_group(mode, k):
#     mode_dir = Path(mode)
#     if not mode_dir.exists():
#         raise FileNotFoundError
#
#     data = []
#     groups = {}
#
#     try:
#         with open(str(mode_dir / f"kmeans={k}.json"), "r") as file:
#             groups = json.load(file)
#
#         return groups
#     except:
#         pass
#
#     with open(str(mode_dir / "file_hash.json"), "r") as file:
#         file_hash = json.load(file)
#
#     with open("file_name.json", "r") as file:
#         file_name = json.load(file)
#
#     for _, j in file_hash.items():
#         binary = [int(c) for c in bin(j)[2:]]
#         while len(binary) != 64:
#             binary = [0] + binary
#         data.append(binary)
#
#     data = np.array(data)
#
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
#
#     for i, j in enumerate(kmeans.labels_):
#         if j in groups.keys():
#             groups[int(j)].append(file_name[str(i)])
#         else:
#             groups[int(j)] = [file_name[str(i)]]
#
#     groups = {k: v for k, v in sorted(groups.items(), key=lambda x: x[0])}
#
#     with open(str(mode_dir / f"kmeans={k}.json"), "w") as file:
#         json.dump(groups, file)
#
#     return groups


def find_name_groups(name_dir):
    pattern = "[ .]"
    groups = {}
    prefix = {}
    num = 0
    
    file_name_path = name_dir + '/file_name.json' 
    name_group_path = name_dir + '/name_groups.json'
    prefix_path = name_dir + '/prefix.json'
    with open(file_name_path, 'r') as file:
        file_name = json.load(file)
        
    for docid, name in file_name.items():
        tokens = re.split(pattern, name)
        if tokens[0] in prefix.keys():
            groups[prefix[tokens[0]]].append(docid)
        else:
            prefix[tokens[0]] = str(num)
            groups[str(num)] = [docid]
            num += 1
    
    with open(name_group_path, "w") as file:
        json.dump(groups, file)
    
    with open(prefix_path, "w") as file:
        json.dump(prefix, file)
    
    return prefix


def find_threshold(name_dir, mode):
    name_groups_path = name_dir + '/name_groups.json'
    similarity_path = name_dir + '/' + mode + '/similarity.json'
    group_similarity_path = name_dir + '/' + mode + '/group_similarity.json'
    with open(name_groups_path, 'r') as file:
        name_groups = json.load(file)
    
    with open(similarity_path, 'r') as file:
        similarity = json.load(file)
    
    similarity_for_each_group = {}
    for group_id, docids in name_groups.items():
        sim = 0
        docids = sorted([int(i) for i in docids])
        if len(docids) == 1:
            similarity_for_each_group[str(group_id)] = 1
            continue
        combs = combinations(docids, 2)
        for pair in combs:
            sim += float(similarity[f'{pair[0]},{pair[1]}'])
        sim /= scipy.special.comb(len(docids), 2, exact=True)
        similarity_for_each_group[str(group_id)] = sim
    
    with open(group_similarity_path, 'w') as file:
        json.dump(similarity_for_each_group, file)
    
    print(sum(similarity_for_each_group.values()) / len(
        similarity_for_each_group))
    
    
def get_precision_recall(name_dir, mode, threshold):
    similarity_path = name_dir + '/' + mode + '/similarity.json'
    file_name_path = name_dir + '/file_name.json'
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    pattern = "[ .]"
    
    with open(similarity_path, 'r') as file:
        similarity = json.load(file)
    
    with open(file_name_path, 'r') as file:
        file_name = json.load(file)
        
    for pair, score in similarity.items():
        score = float(score)
        ids = pair.split(',')
        name1 = file_name[ids[0]]
        name2 = file_name[ids[1]]
        name1 = re.split(pattern, name1)[0]
        name2 = re.split(pattern, name2)[0]
        isClone = name1 == name2
        
        if isClone:
            if score >= threshold:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if score >= threshold:
                false_positive += 1
            else:
                true_negative += 1
                
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return precision, recall

if __name__ == '__main__':
    modes = ["word", "line", "block"]
    android_dir = 'Alloy_models/Android'
    cnf_dir = 'Alloy_models/cnf'
    java_dir = 'Alloy_models/java_file'
    
    alloy = 'alloy_analyzer'
    cnf = 'cnf_analyzer'
    java = 'java_analyzer'
    alloy_analyzer = clone_analyzer("alloy_analyzer", android_dir)
    # alloy_analyzer.read_data_directory()
    # alloy_analyzer.find_file_hashs(modes[1])
    # alloy_analyzer.find_similarity(modes[1])
    # cnf_analyzer = clone_analyzer('cnf_analyzer', cnf_dir)
    # cnf_analyzer.read_data_directory()
    # cnf_analyzer.find_file_hashs(modes[1])
    # cnf_analyzer.find_similarity(modes[1])
    # find_threshold('alloy_analyzer/line')
    # find_threshold('cnf_analyzer/line')
    # find_name_groups('cnf_analyzer')
    # find_threshold('cnf_analyzer', modes[1])

    # alloy_analyzer.find_file_hashs(modes[0])
    # alloy_analyzer.find_similarity(modes[0])

    java_analyzer = clone_analyzer("java_analyzer", java_dir)
    java_analyzer.read_data_directory()
    java_analyzer.find_file_hashs(modes[0])
    java_analyzer.find_similarity(modes[0])

    thresholds = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    precision = defaultdict(list)
    recall = defaultdict(list)
    f1 = defaultdict(list)

    for threshold in thresholds:
        ap, ar = get_precision_recall(alloy, modes[0], threshold)
        cp, cr = get_precision_recall(cnf, modes[1], threshold)
        jp, jr = get_precision_recall(java, modes[0], threshold)
        precision['alloy'].append(ap)
        precision['cnf'].append(cp)
        precision['java'].append(jp)

        recall['alloy'].append(ar)
        recall['cnf'].append(cr)
        recall['java'].append(jr)

        f1['alloy'].append(2 * ap * ar / (ap + ar))
        f1['cnf'].append(2 * cp * cr / (cp + cr))
        f1['java'].append(2 * jp * jr / (jp + jr))
    
    with open('alloy_result', 'w') as file:
        file.write('thresholds,precision,recall,f1\n')
        for i in range(len(thresholds)):
            file.write(f"{thresholds[i]},{precision['alloy'][i]},"
                       f"{recall['alloy'][i]},{f1['alloy'][i]}\n")

    with open('cnf_result', 'w') as file:
        file.write('thresholds,precision,recall,f1\n')
        for i in range(len(thresholds)):
            file.write(f"{thresholds[i]},{precision['cnf'][i]},"
                       f"{recall['cnf'][i]},{f1['cnf'][i]}\n")

    with open('java_result', 'w') as file:
        file.write('thresholds,precision,recall,f1\n')
        for i in range(len(thresholds)):
            file.write(f"{thresholds[i]},{precision['java'][i]},"
                       f"{recall['java'][i]},{f1['java'][i]}\n")

    plt.plot(thresholds, precision['alloy'], label="precision", color="r")
    plt.plot(thresholds, recall['alloy'], label="recall", color="blue")
    plt.plot(thresholds, f1['alloy'], label="f1", color="black")
    plt.title("Alloy")
    plt.legend()
    plt.savefig('Alloy.png')
    plt.clf()

    plt.plot(thresholds, precision['cnf'], label="precision", color="r")
    plt.plot(thresholds, recall['cnf'], label="recall", color="blue")
    plt.plot(thresholds, f1['cnf'], label="f1", color="black")
    plt.title("CNF")
    plt.legend()
    plt.savefig('CNF.png')
    plt.clf()

    plt.plot(thresholds, precision['java'], label="precision", color="r")
    plt.plot(thresholds, recall['java'], label="recall", color="blue")
    plt.plot(thresholds, f1['java'], label="f1", color="black")
    plt.title("java")
    plt.legend()
    plt.savefig('java.png')
    plt.clf()
