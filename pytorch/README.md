# Yelp Sentence Sentiment Analysis

## Deadline: March 17

**install requirements**

python >= 3.6.3

```bash
$ pip install requirements.txt
```

**training and validating models**

SimpleLSTM
```bash
$ python main.py
```

BidirectionalLSTM

```bash
$ python main.py --model bidirectional-lstm
```

**results so far**

| Model             | Training Accuracy | Validation Accuracy |
|-------------------|-------------------|---------------------|
| SimpleLSTM        | 70.88%            | 67.2%               |
| BidirectionalLSTM | 70.75%            | 68.56%              |
| GloVe             |                   |                     |
| Bert              |                   |                     |

**TODO List**
- [X] baseline model
- [X] bigger model
- [ ] better preprocessing
- [ ] class balancing
- [ ] GloVe
- [ ] BERT
