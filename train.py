from argparse import ArgumentParser
from tqdm import tqdm
import dataset
from Vit import Vit
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(train_set, model, optimizer, loss_fn):
    model.train()
    corrects = 0
    total_preds = 0
    for img, label in tqdm(train_set):
        img, label = img.to(device), label.to(device)
        preds = model(img)

        optimizer.zero_grad()
        loss = loss_fn(preds, label)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(F.softmax(preds, dim=-1), dim=-1)
        label = torch.argmax(label, dim=-1)

        corrects += torch.sum(preds == label).item()
        total_preds += preds.shape[0]

    return corrects / total_preds


def test(test_set, model):
    model.eval()
    corrects = 0
    total_preds = 0
    with torch.no_grad():
        for img, label in tqdm(test_set):
            img, label = img.to(device), label.to(device)
            preds = model(img)
            preds = torch.argmax(F.softmax(preds, dim=-1), dim=-1)
            label = torch.argmax(label, dim=-1)

            corrects += torch.sum(preds == label).item()
            total_preds += preds.shape[0]

    return corrects / total_preds

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float) # it seems like 1e-2 is too large
    parser.add_argument('-b', default=1024, type=int)
    parser.add_argument('--epoch', '-e', default=100, type=int)

    args = parser.parse_args()
    
    train_set, test_set = dataset.get_dataset(args.b)
    model = Vit(img_shape=(28, 28)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(args.epoch):
        train_acc = train(train_set, model, optimizer, loss_fn)
        test_acc = test(test_set, model)

        print(f'train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')