from argparse import ArgumentParser
from tqdm import tqdm
import dataset

def train(train_set):
    for img, label in tqdm(train_set):
        print(img.shape, label.shape) 

def test(test_set):
    for img, label in tqdm(test_set):
        print(img.shape, label.shape)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('-b', default=512, type=int)
    parser.add_argument('--epoch', '-e', default=100, type=int)

    args = parser.parse_args()
    
    train_set, test_set = dataset.get_dataset(args.b)

    for e in range(args.epoch):
        train(train_set)
        test(test_set)