import argparse
import os
import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm

from data_loader import get_loader
from model import SimpleLSTMModel, ImprovedLSTMModel

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_data, learning_rate, total_epoch, logging_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(total_epoch):
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

            # if (i + 1) % logging_rate == 0:
            #     train_data.set_description("Epoch {}: Loss: {} Accuracy: {}".format(
            #         epoch, loss, (label == output).float().mean())
            #     )

        # print epoch accuracy
        print("training accuracy: {}/{} ".format(correct, total), correct / total)


def validate(model, valid_data):
    # turn on testing behavior for dropout, batch-norm
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (label, seq, length) in enumerate(tqdm(valid_data)):
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


def predict(model, test_data):
    model.eval()
    prediction = []
    for i, (seq, length) in enumerate(tqdm(test_data)):
        output, length = model(seq, length)
        output = torch.argmax(output) + 1
        prediction.extend(output.cpu())
    return prediction


def main(config):
    # create model directory
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    # load data
    train_data, valid_data, vocab = get_loader(
        fix_length=config.fix_length,
        batch_size=config.batch_size
    )

    # model
    if config.model == "simple-lstm":
        model = SimpleLSTMModel(config.embedding_size,
                                config.hidden_size,
                                config.dropout_rate,
                                vocab).to(device)

    elif config.model == "bidirectional-lstm":
        model = ImprovedLSTMModel(config.embedding_size,
                                  config.hidden_size,
                                  config.dropout_rate,
                                  vocab).to(device)
    print("model loaded...")

    # train
    train(model,
          train_data,
          config.learning_rate,
          config.total_epoch,
          config.logging_rate)

    # save
    torch.save(model.state_dict(), os.path.join(config.model_dir, "{}.pkl".format(config.model)))

    # validate
    validate(model, valid_data)

    # prediction
    # model.load_state_dict(torch.load("{}.pkl".format(config.model))
    # id_list = None
    # test_data = None
    # output = predict(model, test_data)
    # sub_df = pd.DataFrame()
    # sub_df["review_id"] = id_list
    # sub_df["pre"] = output
    # sub_df.to_csv("pre.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # setup parameters
    parser.add_argument("--model", type=str, default="simple-lstm",
                        choices=["simple-lstm", "bidirectional-lstm", "glove-lstm", "bert"])
    parser.add_argument('--data', type=str, default='.root')
    parser.add_argument("--model_dir", type=str, default='./models')
    parser.add_argument("--vocab_path", type=str, default='vocab.pkl')

    # model parameters
    parser.add_argument("--fix_length", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--embedding_size", type=int, default=100)
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