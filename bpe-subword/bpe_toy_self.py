'''Byte Pair Encoding. Use subword representation, which between word and character on scal '''

import re, collections
from typing import Dict, Tuple

def get_pair_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Get count of pairs of consecutive symbools."""

    pairs = {}
    for word, frequency in vocab.items():
        symbools = word.split()

        # count ocurrences of pairs
        for i in range(len(symbools)-1):
            pair = (symbools[i], symbools[i+1])
            current_frequency = pairs.get(pair, 0)
            pairs[pair] = current_frequency + frequency

    return pairs


def merge_vocab(best_pair: Tuple[str, str], vocab_in: Dict[str, int]) -> Dict[str, int]:
    """Merge all ocurrences of the most frequent pair"""

    vocab_out = {}

    pattern = re.escape(' '.join(best_pair))
    replacement = ''.join(best_pair)

    for word_in in vocab_in:
        # replace most frequent pair in all vocabulary
        word_out = re.sub(pattern, replacement, word_in)
        vocab_out[word_out] = vocab_in[word_in]

    return vocab_out


def main():
    vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3,
    'h a p p i e r </w>': 2
    }

    bpe_codes = {}
    num_merges = 10000
    for i in range(num_merges):
        print('\niteration', i)
        pair_stats = get_pair_stats(vocab)
        if not pair_stats:
            break

        best_pair = max(pair_stats, key=pair_stats.get)
        bpe_codes[best_pair] = i

        print('vocab: ', vocab)
        print('best pair: ', best_pair)
        vocab = merge_vocab(best_pair, vocab)

    print('\nfinal vocabulary: ', vocab)
    print('\nbyte pair encoding: ', bpe_codes)

if __name__ == '__main__':
    main()