import re
from collections import defaultdict



'''
text: the text need to be tokenized
mode: "word" tokenized by word
      "line" tokenized by line
      "block" tokenized by block
'''
def tokenize(text: str, mode: str) -> list:
    tokens = []
    if mode == "line":
        pattern = re.compile("^.*?$", re.MULTILINE)
    elif mode == "block":
        pattern = re.compile("^.*?\{.*?\}\n", re.MULTILINE | re.DOTALL)
    else:
        pattern = re.compile("[a-z0-9]+")

    try:
        tokens += re.findall(pattern, text.lower())
    except Exception as err:
        print(err)
    finally:
        return tokens


def compute_word_frequencies(tokens: list) -> dict:
    count = defaultdict(lambda: int())
    try:
        i = 0
        for token in tokens:
            count[token] += 1
            i += 1
    except:
        pass
    finally:
        return count


if __name__ == '__main__':
    # filename = "Alloy_models/synthesized/27 3.als"
    # with open(filename, "r") as file:
    #     token = tokenize(file.read(), "block")
    #
    # for i in token:
    #     print(i)
    
    a = 1
    b = 3
    print(a ^ b)