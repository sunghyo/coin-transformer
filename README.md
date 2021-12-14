# coin-transformer

(Work in progress)

## Prerequisites

```
pip install pyupbit
```

## Datasets

```python
from dataset import UpbitDataset, CoinDataset

upbit_data = UpbitDataset()
train_list, val_list = upbit_data.get_data()

train_dataset = CoinDataset(train_list)
val_dataset = CoinDataset(val_list)
```
