Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
SAVE_DIR = "/content"

if __name__ == "__main__":
    from itertools import product
    import torch

    pp = list(product(range(1,6), repeat=3))
    pp = torch.tensor(pp).float().view(-1, 3)
    cash_bias = torch.ones(size=(pp.shape[0], 1)) * 3.0
    pp = torch.cat([cash_bias, pp], dim=-1)
    pp = torch.softmax(pp, dim=-1)
    pp = pp.numpy()
    print(pp)
    print(pp.shape)