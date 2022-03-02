# Byte Pair Encoding. An efficent rare word encoding technique.
- Note: This repo is heavily borrowed from [offical bpe repo](https://github.com/rsennrich/bpe-subword/).
- Implementation for the paper: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf).
- A basic implementation for bpe, which is based on TDM class PPT, could be found in ```bpe_toy_self.py```.
- A great tutorial toward bpe and Google Sentencepiece Lib introduction is [here](http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html).

## Command fashion usage
More usage details could be found at ```original_README.md```

### 1. Learn bpe
Notice that ```-o``` requires to be a specific file instead of a directory.
```bash
python learn_bpe.py -i "./bpe-subword/tests/data/corpus.en" \
  -o "./bpe-subword/tests/data/bpe.ref" -s 10000 -v
```

### 2. Apply bpe to corpus with the learned 'bpe.ref'
```bash
python apply_bpe.py -i "./bpe-subword/tests/data/corpus.en" \
  --codes "./bpe-subword/tests/data/bpe.ref" \
  -o "./bpe-subword/tests/data/corpus.bpe.en"
```
 \
After applying bpe, a encoded corpus has been generated, available in NLP tasks.
