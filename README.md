## VintageVectors: Unicode Point Tokenization

VintageVectors takes a straightforward, yet unconventional approach to tokenization by representing words as sequences of unicode code points.

```python
# tokens are segmented by whitespace
# num_tokens: max number of words along the sequence(channel) dimension. Also known as context length
# num_chars: max number of characters to represent a word (used 25 in experiments)
tokenized = torch.zeros(num_tokens, num_chars)
for idx, token in enumerate(tokens):
    tokenized[idx] = pad_max(torch.tensor(to_unicode(token)), num_chars)
```

In this method, words are treated as channels/steps, with each word encoded as a sequence of unicode code points in the feature dimension.

### Potential Benefits

The simplicity of this approach could offer benefits in terms of:

- **Memory Efficiency**: By leveraging the channel dimension to store words and the feature dimension to represent the word itself, VintageVectors may provide a memory-friendly solution compared to traditional tokenization methods.

- **Preservation of Word Structure**: Encoding words as sequences of characters preserves the internal structure, which could be advantageous for tasks requiring character-level understanding. Gains benefits from character based tokenization schemes while having a lower number of tokens than Byte Pair Encoding (BPE). 
  
-  **Handle any language out of the box**: Using code points allows us to model any character within unicode.

### Inspiration

This vintage, back-to-basics method draws inspiration from two sources:

1. **Audio and DSP**: Coming from a background in hip hop production and working with raw audio data, representing words as sequences of unicode points feels akin to working with raw samples - the building blocks of rich compositions.

2. **CANINE**: The [CANINE](https://arxiv.org/abs/2103.06874) paper, which tokenizes text at the unicode point level, served as a direct inspiration for this character-level tokenization scheme.

**TODO**
- [ ] Make this repo a pip package
- [ ] Upload text models trained with this tokenization
- [ ] Upload audio models when done training

No claims are made about its effectiveness or superiority. VintageVectors is an exploration of a simple, character-level approach to tokenization, drawing parallels from the worlds of audio and existing methods, while revisiting the foundations of text representations.
