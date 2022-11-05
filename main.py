import generate
import plot
from var import *


def method():
    # 画正常的累积分布分位点图
    new_params = generate.change_normal_params(sample_capacity)
    print(new_params)
    plot.plot_single_normal_cdf(new_params, "theory_cdf.svg")
    plot.plot_single_normal_pdf(new_params, "theory_pdf.svg")
    plot.plot_single_normal_ppf(new_params, "theory_ppf.svg")


# monte carlo计算
def monte_carlo_compute():
    data_result = []
    monte_num.clear()
    for i in range(1, 8):
        i <<= 1
        monte_num.append(1 << i)
    # 生成序列
    for i in range(0, len(monte_num)):
        result = generate.monte_carlo_test(monte_num[i], request_num)
        tup = (monte_num[i], request_num, result)
        data_result.append(tup)
        print("Compute exp", monte_num[i], " Done!")
    plot.plot_monte_carlo_cdf(data_result, "monte_carlo_cdf.svg")
    plot.plot_monte_carlo_ppf(data_result, "monte_carlo_ppf.svg")


# 误差分析
def monte_carlo_error():
    data_result = []
    monte_num.clear()
    for i in range(1, 21):
        monte_num.append(1 << i)
    for i in range(0, len(monte_num)):
        result = generate.monte_carlo_test(monte_num[i], request_num)
        tup = (monte_num[i], request_num, result)
        data_result.append(tup)
        print("Error exp gen", monte_num[i], " Done!")
    plot.plot_monte_carlo_error(data_result, "monte_carlo_exp_time_error.svg")

    # data_result = []
    # request_times = []
    # for i in range(1, 6):
    #     request_times.append(pow(10, i))
    # for i in range(0, len(request_times)):
    #     result = generate.monte_carlo_test(1000, request_times[i])
    #     tup = (1000, request_times[i], result)
    #     data_result.append(tup)
    # plot.plot_monte_carlo_error(data_result, "monte_carlo_request_time_error.svg")


if __name__ == '__main__':
    print("main")

    method()

    monte_carlo_compute()

    monte_carlo_error()
