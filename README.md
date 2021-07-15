# Hand Signs Recognition with Skorch
## Requirements
1. __Build the dataset of size 64x64__: make sure you complete this step before training
```bash
python prepare__dataset.py
```
2. __Base Model experiment__: We created a `base_model` directory under `experiments` directory. 
It contains `params.json` which sets the hyperparameters for the experiments. for e.g.
```json
{
  "learning_rate": 1e-3,
  "batch_size": 32,
  "num_epochs": 10
}
```
For every new experiment, you will need to create a new directory under `experiments` with a similar `params.json` file.

3. __Train__ your experiment. Simply run
```
python train.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the validation set.

## Results
|                          |   accuracy |   precision |   recall |       f1 |
|:-------------------------|-----------:|------------:|---------:|---------:|
| ./experiments/epochs_200 |   0.976852 |    0.975096 | 0.975352 | 0.975157 |
| ./experiments/epochs_100 |   0.972222 |    0.97071  | 0.97048  | 0.970535 |
| ./experiments/epochs_50  |   0.921296 |    0.91821  | 0.921606 | 0.918521 |
| ./experiments/base_model |   0.87037  |    0.865206 | 0.874003 | 0.865857 |