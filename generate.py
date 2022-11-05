import math
import random

import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.stats import zipf
from var import *


# 对数组进行zipf分布加权
def zipf_weight(array):
    frequency_sum = 0.0
    frequency_list = []
    for i in range(0, len(array)):
        rank = i + 1
        frequency = pow(rank, -alpha)
        frequency_sum += frequency
        frequency_list.append(frequency)
    for i in range(0, len(array)):
        array[i] = frequency_list[i] * array[i] / frequency_sum
    return array


# 转换正态分布为加权后的正态分布参数
def change_normal_params(num):
    params = []
    for index in range(0, len(num)):
        ratio = np.full(num[index], 1, dtype=float)
        ratio = zipf_weight(ratio)
        ratio = zipf_weight(ratio)
        sigma_weight = sum(ratio)
        sigma_weight = math.sqrt(sigma_weight)
        tup = (origin_miu, origin_sigma * sigma_weight, num[index])
        params.append(tup)
    return params


def find(array, num):
    left = 0
    right = len(array) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if num >= array[mid]:
            if mid + 1 == len(array):
                return mid
            if num < array[mid + 1]:
                return mid
            left = mid + 1
        else:
            right = mid - 1
    return right


# 生成一组num个数的蒙特卡洛实验有序结果
def monte_carlo_test(times, num):
    video_size = []
    video_cdf = [0]
    results = []
    # 生成一个请求对应到短视频的分布
    ratio = np.full(capacity, 1, dtype=float)
    ratio = zipf_weight(ratio)
    for i in range(1, len(ratio)):
        video_cdf.append(video_cdf[i - 1] + ratio[i - 1])
    for v in range(0, times):
        video_size.clear()
        # 一次蒙特卡洛模拟
        video_size = norm.rvs(loc=origin_miu, scale=origin_sigma, size=capacity).tolist()
        traffic = 0.0
        for request in range(0, num):
            pro = random.random()
            # 寻找此时对应的请求大小
            index = find(video_cdf, pro)
            request_size = video_size[index]
            traffic += request_size
        results.append(traffic / num)
    results.sort()
    return results


# 计算一组蒙特卡洛实验的累计分布函数
def monte_carlo_cdf(result):
    for index in range(0, len(result)):
        # 给定一个上届看有多少个
        print("1")


# 计算一组蒙特卡洛实验的分位函数
def monte_carlo_ppf(result):
    for index in range(0, len(result)):
        # 给定一个概率
        print("2")
