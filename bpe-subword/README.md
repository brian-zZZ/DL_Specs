# Byte Pair Encoding. Efficent rare word encoding technique.
- Note: This repo is heavily borrowed from [offical bpe repo](https://github.com/rsennrich/subword-nmt/).
- Implementation for paper: Neural Machine Translation of Rare Words with Subword Units(https://arxiv.org/pdf/1508.07909.pdf).
- A basic implementation for bpe, which is based on TDM class PPT, could be found in ```bpe_toy_self.py```.
- A great tuitoral and Google Sentencepiece Lib introduction is here(http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html).

## Command fashion usage
More uage details could be found at ```original_README.md```

### Learn bpe
Notice that ```-o``` requires to be a specific file instead of a directory.
```bash
python learn_bpe.py -i "./subword-nmt/tests/data/corpus.en" \
  -o "./subword-nmt/tests/data/bpe.ref" -s 10000 -v
```

### Apply bpe to corpus with the learned 'bpe.ref'
```bash
python apply_bpe.py -i "./subword-nmt/tests/data/corpus.en" \
  --codes "./subword-nmt/tests/data/bpe.ref" \
  -o "./subword-nmt/tests/data/corpus.bpe.en"
```
 \
After applying bpe, a encoded corpus has been generated, available in NLP tasks.