## Learn bpe
Notice that ```-o``` requires to be a specific file instead of a directory.
```bash
python learn_bpe.py -i "D:/DL_Impls/subword-nmt/subword_nmt/tests/data/corpus.en" \
  -o "D:/DL_Impls/subword-nmt/subword_nmt/tests/data/bpe.ref" -s 10000 -v
```

## Apply bpe to corpus with the learned 'bpe.ref'
```bash
python apply_bpe.py -i "D:/DL_Impls/subword-nmt/subword_nmt/tests/data/corpus.en" \
  --codes "D:/DL_Impls/subword-nmt/subword_nmt/tests/data/bpe.ref" \
  -o "D:/DL_Impls/subword-nmt/subword_nmt/tests/data/corpus.bpe.en"
```