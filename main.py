import argparse
import os
import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm, trange

from data_loader import get_loader
from model import SimpleLSTMModel, ImprovedLSTMModel, GloveModel, AWDModel

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_validate(model, model_path, train_data, valid_data, learning_rate, total_epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    cycles = 0
    for epoch in trange(total_epoch):
        correct = 0
        total = 0
        train_data = tqdm(train_data)
        model.train()
        if cycles == 3:
            print("switching to ASGD")
            optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

        for i, (label, seq, length) in enumerate(train_data):
            # sort by descending order for packing
            length, permute = length.sort(dim=0, descending=True)
            seq = seq[permute]
            label = label[permute]

            # convert to cuda
            seq = seq.to(device)
            label = label.to(device)

            # Forward, backward, and optimize
            output = model(seq, length)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = torch.argmax(output, 1)
            correct += (output == label).sum().item()
            total += output.size(0)

        # print epoch accuracy
        print("training accuracy: {}/{} ".format(correct, total), correct / total)
        valid_acc = validate(model, valid_data)

        # keep best acc model
        if valid_acc > best_acc:
            torch.save(model.state_dict(), model_path)
            best_acc = valid_acc
            cycles = 0
        else:
            cycles += 1


def train(model, train_data, learning_rate, total_epoch, logging_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in trange(total_epoch):
        correct = 0
        total = 0
        train_data = tqdm(train_data)
        for i, (label, seq, length) in enumerate(train_data):
            # sort by descending order for packing
            length, permute = length.sort(dim=0, descending=True)
            seq = seq[permute]
            label = label[permute]

            # convert to cuda
            seq = seq.to(device)
            label = label.to(device)

            # Forward, backward, and optimize
            output = model(seq, length)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = torch.argmax(output, 1)
            correct += (output == label).sum().item()
            total += output.size(0)

        # print epoch accuracy
        print("training accuracy: {}/{} ".format(correct, total), correct / total)


def validate(model, valid_data):
    # turn on testing behavior for dropout, batch-norm
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (label, seq, length) in enumerate(valid_data):
            # sort by descending order for packing
            length, permute = length.sort(dim=0, descending=True)
            seq = seq[permute]
            label = label[permute]

            # convert to cuda
            seq = seq.to(device)
            label = label.to(device)

            # Forward
            output = model(seq, length)
            output = torch.argmax(output, 1)
            correct += (output == label).sum().item()
            total += output.size(0)
    print("validation accuracy: {}/{} ".format(correct, total), correct / total)
    return correct / total


def predict(model, test_data):
    model.eval()
    prediction = []
    for i, (_, seq, length) in enumerate(tqdm(test_data)):
        # sort by descending order for packing
        length, sort_indices = length.sort(dim=0, descending=True)
        _, unsort_indices = torch.sort(sort_indices, dim=0)
        seq = seq[sort_indices]

        # convert to cuda
        seq = seq.to(device)

        output = model(seq, length)
        output = torch.argmax(output, 1) + 1
        # unsort output
        output = output.cpu().index_select(0, unsort_indices)
        prediction.extend(output.tolist())

    return prediction


def main(config):
    # create model directory
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    # load data
    train_data, valid_data, test_data, vocab = get_loader(
        fix_length=config.fix_length,
        vocab_threshold=config.vocab_threshold,
        batch_size=config.batch_size
    )

    # model
    if config.model == "best":
        fasttext = vocab.get_embedding('fasttext', 300)
        model = AWDModel(300,
                         config.hidden_size,
                         config.dropout_rate,
                         fasttext).to(device)

    elif config.model == "simple-lstm":
        model = SimpleLSTMModel(config.embedding_size,
                                config.hidden_size,
                                config.dropout_rate,
                                vocab).to(device)

    elif config.model == "bidirectional-lstm":
        model = ImprovedLSTMModel(config.embedding_size,
                                  config.hidden_size,
                                  config.dropout_rate,
                                  vocab).to(device)

    elif config.model == 'glove-lstm':
        glove = vocab.get_embedding('glove', config.embedding_size)
        model = GloveModel(config.embedding_size,
                           config.hidden_size,
                           config.dropout_rate,
                           glove).to(device)

    elif config.model == 'awd-lstm':
        fasttext = vocab.get_embedding('fasttext', 300)
        model = AWDModel(300,
                         config.hidden_size,
                         config.dropout_rate,
                         fasttext).to(device)

    print("model loaded...")

    train_and_validate(model,
                       os.path.join(config.model_dir, "{}.pkl".format(config.model)),
                       train_data,
                       valid_data,
                       config.learning_rate,
                       config.total_epoch)

    # model.load_state_dict(torch.load(os.path.join(config.model_dir, "{}.pkl".format(config.model))))

    output = predict(model, test_data)
    sub_df = pd.DataFrame()
    sub_df["review_id"] = test_data.dataset.df['review_id']
    sub_df["pre"] = output
    sub_df["text"] = test_data.dataset.df['text']
    sub_df.to_csv("pre.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # setup parameters
    parser.add_argument("--model", type=str, default="best",
                        choices=["best", "simple-lstm", "bidirectional-lstm", "glove-lstm", "awd-lstm"])
    parser.add_argument('--data', type=str, default='.root')
    parser.add_argument("--model_dir", type=str, default='./models')
    parser.add_argument("--vocab_path", type=str, default='vocab.pkl')

    # pre-processing parameters
    parser.add_argument("--vocab_threshold", type=int, default=1)

    # model parameters
    parser.add_argument("--fix_length", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--embedding_size", type=int, default=100, choices=[100, 300])
    parser.add_argument("--dropout_rate", type=float, default=0.5)

    # training parameters
    parser.add_argument("--total_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--logging_rate", type=int, default=100)
    args = parser.parse_args()

    print(args)
    print('device: ', device)

    main(args)
