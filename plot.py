import math

import matplotlib
import numpy
import numpy as np
import generate
from matplotlib import pyplot
from scipy.stats import norm
from var import *

pyplot.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 汉字防止出现乱码
pyplot.rcParams['axes.unicode_minus'] = False
palette = ['#00FFFF',
           '#0000FF', '#8A2BE2',
           '#7FFF00', '#D2691E', '#DC143C',
           '#006400', '#BDB76B', '#8B008B',
           '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B',
           '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF',
           '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF', '#FFD700',
           '#DAA520', '#808080', '#008000', '#ADFF2F', '#F0FFF0', '#FF69B4', '#CD5C5C',
           '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00', '#FFFACD',
           '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#90EE90', '#D3D3D3', '#FFB6C1',
           '#FFA07A', '#20B2AA', '#87CEFA', '#778899', '#B0C4DE', '#FFFFE0', '#00FF00',
           '#32CD32', '#FAF0E6', '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3',
           '#9370DB', '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#191970',
           '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000',
           '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE',
           '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD', '#B0E0E6',
           '#800080', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#FAA460',
           '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD', '#708090',
           '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347',
           '#40E0D0', '#EE82EE', '#F5DEB3', '#FFFFFF', '#F5F5F5', '#FFFF00', '#9ACD32']
markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']


# 绘制随样本数量变化的一组参数的正态分布概率密度函数
def plot_single_normal_pdf(params, filename):
    pyplot.figure()
    left = float('inf')
    right = float('-inf')
    for index in range(0, len(params)):
        miu = params[index][0]
        sigma = params[index][1]
        now_right = miu + 2 * sigma
        now_left = miu - 2 * sigma
        if now_left < left:
            left = now_left
        if now_right > right:
            right = now_right
    abscissa = numpy.linspace(math.floor(left), math.ceil(right), 1000)
    for index in range(0, len(params)):
        miu = params[index][0]
        sigma = params[index][1]
        cap = sample_capacity[index]
        ordinate = norm.pdf(abscissa, miu, sigma)
        pyplot.plot(abscissa, ordinate, color=palette[index], label=f'S={cap}', linewidth=0.6)
    pyplot.xlabel("T(x)统计量(MB)")
    pyplot.ylabel("概率")
    pyplot.title("T(x)概率密度函数" + f'(X~N({origin_miu},{origin_sigma * origin_sigma}))')
    pyplot.legend(loc=1)
    pyplot.savefig(photo_dir + filename, format='svg')
    pyplot.show()


# 绘制随样本数量变化的一组参数的正态分布累计分布函数
def plot_single_normal_cdf(params, filename):
    pyplot.figure()
    left = float('inf')
    right = float('-inf')
    for index in range(0, len(params)):
        miu = params[index][0]
        sigma = params[index][1]
        now_right = miu + 2 * sigma
        now_left = miu - 2 * sigma
        if now_left < left:
            left = now_left
        if now_right > right:
            right = now_right
    abscissa = numpy.linspace(math.floor(left), math.ceil(right), 1000)
    for index in range(0, len(params)):
        miu = params[index][0]
        sigma = params[index][1]
        cap = sample_capacity[index]
        ordinate = norm.cdf(abscissa, miu, sigma)
        pyplot.plot(abscissa, ordinate, color=palette[index], label=f'S={cap}', linewidth=0.6)
    pyplot.xlabel("T(x)统计量(MB)")
    pyplot.ylabel("累计概率")
    pyplot.title("α(t)累积分布函数" + f'(X~N({origin_miu},{origin_sigma * origin_sigma}))')
    pyplot.legend(loc=2)
    pyplot.savefig(photo_dir + filename, format='svg')
    pyplot.show()


# 绘制随样本数量变化的一组参数的正态分布分位点函数
def plot_single_normal_ppf(params, filename):
    pyplot.figure()
    left = 0
    right = 1
    abscissa = numpy.linspace(left, right, 1000)
    for index in range(0, len(params)):
        miu = params[index][0]
        sigma = params[index][1]
        cap = sample_capacity[index]
        ordinate = norm.ppf(abscissa, miu, sigma)
        pyplot.plot(abscissa, ordinate, color=palette[index], label=f'S={cap}', linewidth=0.6)
    pyplot.xlabel("概率")
    pyplot.ylabel("T(x)统计量(MB)")
    pyplot.title("t(α)分位点函数" + f'(X~N({origin_miu},{origin_sigma * origin_sigma}))')
    pyplot.legend(loc=2)
    pyplot.savefig(photo_dir + filename, format='svg')
    pyplot.show()


# 绘制蒙特卡洛不同数量得出的累积分布函数
def plot_monte_carlo_cdf(data, filename):
    pyplot.figure()
    # 理论计算分布参数
    ratio = np.full(capacity, 1, dtype=float)
    ratio = generate.zipf_weight(ratio)
    ratio = generate.zipf_weight(ratio)
    sigma_weight = sum(ratio)
    sigma_weight = math.sqrt(sigma_weight)
    new_sigma = sigma_weight * origin_sigma
    left = origin_miu - 2 * new_sigma
    right = origin_miu + 2 * new_sigma
    # 理论计算概率
    abscissa = numpy.linspace(math.floor(left), math.ceil(right), 1000)
    normal_ordinate = norm.cdf(abscissa, origin_miu, new_sigma)
    pyplot.plot(abscissa, normal_ordinate, color="black", label=f'理论计算', linewidth=0.5)
    # 一系列模拟结果
    for result_index in range(0, len(data)):
        exp_times = data[result_index][0]
        result = data[result_index][2]
        result_count = len(result)
        ordinate = []
        for abscissa_index in range(0, len(abscissa)):
            # 计算小于上届的统计量频率
            if abscissa[abscissa_index] < result[0]:
                count = 0
            else:
                count = generate.find(result, abscissa[abscissa_index])
            ordinate.append(count / result_count)
        pyplot.plot(abscissa, ordinate, color=palette[result_index], label=f'实验次数={exp_times}', linewidth=0.5)
    pyplot.xlabel("T(x)统计量(MB)")
    pyplot.ylabel("累计概率")
    pyplot.title("Monte-Carlo计算α(t)累积分布函数与理论计算对比")
    pyplot.legend(loc=2)
    pyplot.savefig(photo_dir + filename, format='svg')
    pyplot.show()


# 绘制蒙特卡洛不同数量得出的分位点函数
def plot_monte_carlo_ppf(data, filename):
    pyplot.figure()
    left = 0
    right = 1
    abscissa = numpy.linspace(left, right, 1000)
    # 理论计算值
    ratio = np.full(capacity, 1, dtype=float)
    ratio = generate.zipf_weight(ratio)
    ratio = generate.zipf_weight(ratio)
    sigma_weight = sum(ratio)
    sigma_weight = math.sqrt(sigma_weight)
    normal_ordinate = norm.ppf(abscissa, origin_miu, origin_sigma * sigma_weight)
    pyplot.plot(abscissa, normal_ordinate, color='black', label="理论计算", linewidth=0.5)
    for result_index in range(0, len(data)):
        exp_times = data[result_index][0]
        result = data[result_index][2]
        result_count = len(result)
        ordinate = []
        for abscissa_index in range(0, len(abscissa)):
            # 计算monte carlo每一个的分位点
            quantile = math.floor(result_count * abscissa[abscissa_index])
            if quantile == result_count:
                quantile -= 1
            ordinate.append(result[quantile])
        pyplot.plot(abscissa, ordinate, color=palette[result_index], label=f'实验次数={exp_times}', linewidth=0.5)
    pyplot.xlabel("概率")
    pyplot.ylabel("T(x)统计量(MB)")
    pyplot.title("Monte-Carlo计算t(α)分位点函数与理论计算对比")
    pyplot.legend(loc=2)
    pyplot.savefig(photo_dir + filename, format='svg')
    pyplot.show()


# 绘制蒙特卡洛误差得出的分位点函数
def plot_monte_carlo_error(data, filename):
    fig, ax1 = pyplot.subplots()
    left = 0
    right = 1
    abscissa = numpy.linspace(left, right, 1000)
    # 理论计算值
    ratio = np.full(capacity, 1, dtype=float)
    ratio = generate.zipf_weight(ratio)
    ratio = generate.zipf_weight(ratio)
    sigma_weight = sum(ratio)
    sigma_weight = math.sqrt(sigma_weight)
    normal_ordinate = norm.ppf(abscissa, origin_miu, origin_sigma * sigma_weight)
    x_values = []
    rate_values = []
    square_values = []
    for result_index in range(0, len(data)):
        exp_times = data[result_index][0]
        request_times = data[result_index][1]
        x_values.append(exp_times)
        result = data[result_index][2]
        result_count = len(result)
        sum_rate = 0.0
        sum_square = 0.0
        # 计算单次蒙特卡洛均方误差
        for abscissa_index in range(1, len(abscissa) - 1):
            real_value = normal_ordinate[abscissa_index]
            quantile = math.floor(result_count * abscissa[abscissa_index])
            if quantile == result_count:
                quantile -= 1
            sub = abs(result[quantile] - real_value)
            rate = sub / real_value
            square = sub * sub
            sum_rate += rate
            sum_square += square
        sum_rate /= len(abscissa) - 2
        sum_rate = sum_rate * 100
        sum_square /= len(abscissa) - 2
        rate_values.append(sum_rate)
        square_values.append(sum_square)
        print(exp_times, request_times, sum_rate, sum_square)

    ax2 = ax1.twinx()
    ax1.plot(x_values, rate_values, color="red", label="平均相对误差比例")
    ax2.plot(x_values, square_values, color="blue", label="均方误差")
    ax1.set_yscale('log')
    # ax1.set_ylim([1e-2, 1e1])
    ax1.set_xscale('log')
    ax2.set_yscale('log')
    ax1.set_xlabel("实验次数")
    ax1.set_ylabel("误差比例均值(%)")
    ax2.set_ylabel("误差平方均值[(MB)²]")
    pyplot.title("Monte-Carlo与理论计算的误差分析")
    ax1.legend(loc=3)
    ax2.legend(loc=1)
    pyplot.savefig(photo_dir + filename, format='svg')
    pyplot.show()
