import os

import fire
import logging
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EpochScoring, TrainEndCheckpoint
from skorch.cli import parse_args
from skorch.helper import predefined_split
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

import utils

def seed_everything(SEED=212):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

class SIGNSDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image = Image.open(self.filenames[item])
        image = self.transform(image)
        return image, self.labels[item]

    def targets(self):
        return self.labels

def load_dataset(data_dir='./dataset/64x64_SIGNS'):
    train_transform = transforms.Compose([
        # transforms.Resize(64),            # Remove if the image already 64x64 size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        # transforms.Resize(64),              # Remove if the image already 64x64 size
        transforms.ToTensor()
    ])
    train_ds = SIGNSDataset(os.path.join(data_dir, 'train_signs'), train_transform)
    val_ds = SIGNSDataset(os.path.join(data_dir, 'val_signs'), val_transform)
    test_ds = SIGNSDataset(os.path.join(data_dir, 'test_signs'), val_transform)

    return train_ds, val_ds, test_ds

class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:
        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(8*8*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 6)
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .
        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8

        # flatten the output for each image
        s = s.view(-1, 8*8*self.num_channels*4)             # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 6

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)

def load_model():
    model = nn.Sequential(
        nn.Conv2d(3, 3, 3, stride=1, padding=1),
        nn.BatchNorm2d(3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d()
    )
    return model

def get_trainer(experiment_dir, model, loss_fn, optim_fn, hyperparameters):
    checkpoint = Checkpoint(fn_prefix='best_', monitor='valid_acc_best',
                            dirname=experiment_dir)
    last_train_cp = TrainEndCheckpoint(dirname=experiment_dir)
    acc = EpochScoring(scoring='accuracy', on_train=True, name='train_acc')
    net = NeuralNetClassifier(
        model,
        criterion=loss_fn,
        lr=hyperparameters.learning_rate,
        batch_size=hyperparameters.batch_size,
        max_epochs=hyperparameters.num_epochs,
        optimizer=optim_fn,
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=4,
        callbacks=[
            checkpoint, last_train_cp,
            acc,
        ],
        device='cuda'
    )
    return net

def train(experiment_dir = './experiments/base_model'):
    """Training SIGN Classification Model"""
    seed_everything(SEED=212)
    utils.set_logger(os.path.join(experiment_dir, 'train.log'))

    params_json = os.path.join(experiment_dir, 'params.json')
    assert os.path.isfile(params_json), f'No json configuration file found at {params_json}'
    hyperparameters = utils.load_hyperparams(params_json)
    train_ds, val_ds, test_ds = load_dataset()
    loss_fn = nn.CrossEntropyLoss
    optim_fn = torch.optim.SGD
    model = Net(hyperparameters)

    net = get_trainer(experiment_dir,model, loss_fn, optim_fn, hyperparameters)
    net.set_params(train_split=predefined_split(val_ds))
    net.fit(train_ds, y=None)
    y_pred = net.predict(val_ds)
    metrics = {
        'accuracy': accuracy_score(y_pred, val_ds.targets()),
        'precision': precision_score(y_pred, val_ds.targets(), average='macro'),
        'recall': recall_score(y_pred, val_ds.targets(), average='macro'),
        'f1': f1_score(y_pred, val_ds.targets(), average='macro')
    }
    # print(metrics)
    metrics_file_path = os.path.join(experiment_dir, 'metrics_val_last_weight.json')
    logging.info(f'Save Validation metrics to {metrics_file_path}')
    utils.save_dict_to_json(metrics, metrics_file_path)

def evaluate(experiment_dir = './experiments/base_model'):
    """Evaluate SIGN Classification Model"""
    seed_everything(SEED=212)

    params_json = os.path.join(experiment_dir, 'params.json')
    assert os.path.isfile(params_json), f'No json configuration file found at {params_json}'
    hyperparameters = utils.load_hyperparams(params_json)

    train_ds, val_ds, test_ds = load_dataset()
    loss_fn = nn.CrossEntropyLoss
    optim_fn = torch.optim.SGD
    model = Net(hyperparameters)

    utils.set_logger(os.path.join(experiment_dir, 'train.log'))

    net = get_trainer(experiment_dir, model, loss_fn, optim_fn, hyperparameters)
    net.initialize()
    checkpoint = Checkpoint(fn_prefix='best_', monitor='valid_acc_best',
                            dirname=experiment_dir)
    net.load_params(checkpoint=checkpoint)
    y_pred = net.predict(val_ds)
    metrics = {
        'accuracy': accuracy_score(y_pred, val_ds.targets()),
        'precision': precision_score(y_pred, val_ds.targets(), average='macro'),
        'recall': recall_score(y_pred, val_ds.targets(), average='macro'),
        'f1': f1_score(y_pred, val_ds.targets(), average='macro')
    }

    metrics_file_path = os.path.join(experiment_dir, 'metrics_val_best_weight.json')
    logging.info(f'Save Validation metrics to {metrics_file_path}')
    utils.save_dict_to_json(metrics, metrics_file_path)

    y_pred = net.predict(test_ds)
    metrics = {
        'accuracy': accuracy_score(y_pred, test_ds.targets()),
        'precision': precision_score(y_pred, test_ds.targets(), average='macro'),
        'recall': recall_score(y_pred, test_ds.targets(), average='macro'),
        'f1': f1_score(y_pred, test_ds.targets(), average='macro')
    }
    metrics_file_path = os.path.join(experiment_dir, 'metrics_test_best_weight.json')
    logging.info(f'Save Test metrics to {metrics_file_path}')
    utils.save_dict_to_json(metrics, metrics_file_path)

def train_and_evaluate(experiment_dir = './experiments/base_model'):
    """Train and evaluate the model"""
    train(experiment_dir)
    evaluate(experiment_dir)

if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'evaluate': evaluate,
        'train_and_evaluate': train_and_evaluate,
    })