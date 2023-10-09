from argparse import ArgumentParser
from tqdm import tqdm
import dataset
from ViT import ViT
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'using device {device}')

def train(train_set, model, optimizer, loss_fn):
    model.train()
    y_true = []
    y_pred = []
    for img, labels in tqdm(train_set):
        img, labels = img.to(device), labels.to(device)
        preds = model(img)

        optimizer.zero_grad()
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()


        preds = torch.argmax(F.softmax(preds, dim=-1), dim=-1).tolist()
        labels = torch.argmax(labels, dim=-1).tolist()

        y_true += labels
        y_pred += preds

    return y_true, y_pred


def test(test_set, model):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for img, labels in tqdm(test_set):
            img, labels = img.to(device), labels.to(device)
            preds = model(img)
            preds = torch.argmax(F.softmax(preds, dim=-1), dim=-1).tolist()
            labels = torch.argmax(labels, dim=-1).tolist()

            y_true += labels
            y_pred += preds

    return y_true, y_pred

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float) # the model is learning-rate sensitive, use smaller learning rate 
    parser.add_argument('--batch_size', '-b', default=2048, type=int)
    parser.add_argument('--epoch', '-e', default=80, type=int)
    parser.add_argument('--patch_size', '-p', default=7, type=int)
    parser.add_argument('--embedded_dim', '-d', default=768, type=int)
    parser.add_argument('--encoder_layer', '-l', default=8, type=int)
    parser.add_argument('--num_class', '-c', default=10, type=int)
    parser.add_argument('--num_head', '-nh', default=8, type=int)
    parser.add_argument('--drop', default=0.1, type=float)
    parser.add_argument('--dataset', '-ds', default='mnist', type=str, help='Currently support only MNIST(mnist) and CIFAR-10(cifar)')
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float)

    args = parser.parse_args()
    
    train_set, test_set, img_shape = dataset.get_dataset(args.batch_size, args.dataset)
    model = ViT(
        img_shape=img_shape,
        patch_size=args.patch_size,
        embedded_dim=args.embedded_dim,
        encoder_layers=args.encoder_layer,
        num_class=args.num_class,
        num_head=args.num_head,
        drop=args.drop
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(args.epoch):
        print(f'epoch: {e}')

        train_true, train_preds = train(train_set, model, optimizer, loss_fn)
        test_true, test_preds = test(test_set, model)

        print(f'train_acc: {accuracy_score(train_true, train_preds):.4f}, test_acc: {accuracy_score(test_true, test_preds):.4f}')

            