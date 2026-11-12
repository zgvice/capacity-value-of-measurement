"""原15分钟负荷协方差；各实验共用相同计算。"""
import numpy as np
from scipy.linalg import toeplitz
DAY=96  # 每日96个15分钟点。

def load_covariance(c):
    # 沿用原负荷生成：日共同项与日内AR项；不改变相关速度。
    return (c.load_daily_variance_share
            + (1-c.load_daily_variance_share)
            * toeplitz(c.load_rho15 ** np.arange(DAY)))


def block_covariance(operator, daily):
    covariance = np.zeros((operator.shape[0], operator.shape[0]))
    for day in np.unique(operator.indices // DAY):
        part = operator[:, day*DAY:(day+1)*DAY]
        rows = np.unique(part.nonzero()[0])
        block = part[rows].toarray()
        covariance[np.ix_(rows, rows)] += block @ daily @ block.T
    return covariance

