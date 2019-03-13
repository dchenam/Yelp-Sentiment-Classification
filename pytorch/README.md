# Yelp Sentence Sentiment Analysis

## Deadline: March 17

**install requirements**

python >= 3.6.3

```bash
$ pip install requirements.txt
```

**training and validating models**

AWD-LSTM

```bash
$ python main.py --total_epochs 20
```

SimpleLSTM
```bash
$ python main.py --model simple-lstm
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
| AWD-LSTM          | 72.85%            | 69.37%              |
| Bert              |                   |                     |

**TODO List**
- [X] baseline model
- [X] bigger model
- [X] better preprocessing
- [X] class balancing
- [X] GloVe
- [ ] BERT
