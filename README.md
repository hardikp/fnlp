# fnlp

This repo contains scripts to gather finance data and train NLP models using the text data.

## Word Vectors

Trained word vectors are available on the [releases](https://github.com/hardikp/fnlp/releases) page.

Let's check if the closest words make sense.

```bash
$ python3 test_word_vectors.py --word IRA
Roth
SEP
IRAs
401
retirement

$ python3 test_word_vectors.py --word option
call
put
options
exercise
underlying

$ python3 test_word_vectors.py --word stock
shares
market
stocks
share
price
```
