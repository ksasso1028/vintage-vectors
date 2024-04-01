import torch
import random



def tokenize(text,num_chars=25, num_tokens=300):
    # tokenize words by whitespace
    tokens = text.split()
    tokenized = torch.zeros(num_tokens ,num_chars)
    for token in range(len(tokens)):
        tokenized[token] = pad_max(torch.tensor(to_unicode(tokens[token])).float(), num_chars)
    return tokenized


def to_unicode(text, tensor=False):
    code_points = [ord(char) for char in text]
    if tensor:
        code_points = torch.tensor(code_points).float()
    return code_points

def to_text(code_points):
    text = [chr(point) for point in code_points]
    return text


def pad_max(text, max=1000):
    #print(len(text))
    if len(text) < max:
        pad = (0, (max - len(text)))
    # print(pad)
        text = torch.nn.functional.pad(text.clone(), pad)
    else:
        text = text[:max]
    return text

# data augmentation
def shift_list(lst, n):
    if n == 0:
        return lst
    elif n > 0:
        return [0] * n + lst
    else:
        return lst[-n:] + [0] * (len(lst) + n)