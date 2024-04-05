import torch

def tokenize(text, max_word_length=25, max_tokens=300):
    """
    Tokenizes input text into words and encodes each word into a tensor representation.

    Args:
        text (str): Input text to be tokenized.
        max_word_length (int, optional): Maximum number of characters for each word tensor. Defaults to 25.
        max_tokens (int, optional): Maximum number of tokens in the output tensor. Defaults to 300.

    Returns:
        torch.Tensor: Tensor representation of tokenized text.
    """
    tokens = text.split()
    tokenized = torch.zeros(max_tokens, max_word_length)
    for idx, token in enumerate(tokens):
        if idx < max_tokens:
            tokenized[idx] = pad_tensor(to_unicode(token), max_word_length)
    return tokenized

def to_unicode(text):
    """
    Converts input text into Unicode code points.

    Args:
        text (str): Input text to be converted.

    Returns:
        torch.Tensor: Tensor of Unicode code points.
    """
    return torch.tensor([ord(char) for char in text]).float()

def pad_tensor(tensor, length):
    """
    Pads input tensor with zeros or truncates it to match the specified length.

    Args:
        tensor (torch.Tensor): Input tensor to be padded or truncated.
        length (int): Desired length for the output tensor.

    Returns:
        torch.Tensor: Padded or truncated tensor.
    """
    return torch.nn.functional.pad(tensor[:length], (0, max(0, length - len(tensor))))
