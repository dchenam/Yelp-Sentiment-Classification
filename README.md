# Yelp Sentence Sentiment Analysis


**install requirements**

python >= 3.6.3

```bash
$ pip install requirements.txt
```

**dataset**

- used the 100k subset of the Yelp Dataset


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

| Model                    | Training Accuracy | Validation Accuracy |
|--------------------------|-------------------|---------------------|
| SimpleLSTM               | 70.88%            | 67.2%               |
| BidirectionalLSTM        | 70.75%            | 68.56%              |
| AWD-LSTM (no pretraining)| 72.85%            | 69.37%              |
