from argparse import ArgumentParser
from tqdm import tqdm
import dataset
from ViT import ViT
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchsummary import summary

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
    parser.add_argument('--batch_size', '-b', default=512, type=int)
    parser.add_argument('--epoch', '-e', default=500, type=int)
    parser.add_argument('--patch_size', '-p', default=7, type=int)
    parser.add_argument('--embedded_dim', '-d', default=384, type=int)
    parser.add_argument('--encoder_layer', '-l', default=6, type=int)
    parser.add_argument('--num_class', '-c', default=10, type=int)
    parser.add_argument('--num_head', '-nh', default=8, type=int)
    parser.add_argument('--drop', default=0.05, type=float)
    parser.add_argument('--dataset', '--ds', default='mnist', type=str, help='Currently support only MNIST(mnist) and CIFAR-10(cifar)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--label_smoothing', '--ls', default=0, type=float)

    args = parser.parse_args()
    
    train_set, test_set, img_shape = dataset.get_dataset(args.batch_size, args.dataset)

    _, h, w = img_shape
    assert h % args.patch_size == 0 and w % args.patch_size == 0, ('The shape of image is not divisible by the size of patch. This might lead to the lost of information from the image.')

    model = ViT(
        img_shape=img_shape,
        patch_size=args.patch_size,
        embedded_dim=args.embedded_dim,
        encoder_layers=args.encoder_layer,
        num_class=args.num_class,
        num_head=args.num_head,
        drop=args.drop
    ).to(device)


    summary(model, input_size=img_shape)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = 0

    for e in range(args.epoch):
        print(f'epoch: {e}')

        train_true, train_preds = train(train_set, model, optimizer, loss_fn)
        test_true, test_preds = test(test_set, model)

        train_acc = accuracy_score(train_true, train_preds)
        test_acc = accuracy_score(test_true, test_preds)

        print(f'train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')

        if test_acc > best_acc:
            best_acc = test_acc
            print('saving model...')
            torch.save(model.state_dict(), f'{args.dataset}.pth')

            