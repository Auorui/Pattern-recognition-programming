# @Auorui(夏天是冰红茶)
import numpy as np
from scipy.stats import norm


class Datasets:
    # 一个简单的数据加载器
    def __init__(self, datapath, t):
        self.datapath = datapath
        self.data = np.loadtxt(self.datapath)  # 二维数组
        self.height = self.data[:, 0]
        self.weight = self.data[:, 1]
        self.length = len(self.data)
        self.t = t

    def __len__(self):
        return self.length

    def mean(self, data):
        # 均值,可以使用np.mean替换
        total = 0
        for x in data:
            total += x
        return total / self.length

    def var(self, data):
        # 方差,可以使用np.var替换
        mean = self.mean(data)
        sq_diff_sum = 0
        for x in data:
            diff = x - mean
            sq_diff_sum += diff ** 2
        return sq_diff_sum / self.length

    def retain(self, *args):
        # 保留小数点后几位
        formatted_args = [round(arg, self.t) for arg in args]
        return tuple(formatted_args)

    def __call__(self):
        mean_height = self.mean(self.height)
        var_height = self.var(self.height)
        mean_weight = self.mean(self.weight)
        var_weight = self.var(self.weight)
        return self.retain(mean_height, var_height, mean_weight, var_weight)

    def calculate_bmi(self):
        # 计算BMI作为新特征
        height_meters = self.height / 100  # 将身高从厘米转换为米
        bmi = self.weight / (height_meters ** 2)  # BMI计算公式
        return bmi

def Dataloader(maledata,femaledata):
    mmh, mvh, mmw, mvw = maledata()
    fmh, fvh, fmw, fvw = femaledata()

    male_height_dist = norm(loc=mmh, scale=mvh**0.5)
    male_weight_dist = norm(loc=mmw, scale=mvw**0.5)
    female_height_dist = norm(loc=fmh, scale=fvh**0.5)
    female_weight_dist = norm(loc=fmw, scale=fvw**0.5)

    data_dist = {
        'mh': male_height_dist,
        'mw': male_weight_dist,
        'fh': female_height_dist,
        'fw': female_weight_dist
    }

    return data_dist


def classify(height=None, weight=None, ways=1):
    """
    根据身高、体重或身高与体重的方式对性别进行分类

    :param height: 身高
    :param weight: 体重
    :param ways: 1 - 采用身高
                 2 - 采用体重
                 3 - 采用身高与体重
    :return: 'Male' 或 'Female'，表示分类结果
    """
    # 先验概率的公式 : P(w1) = m1 / m ,样本总数为m,属于w1类别的有m1个样本。

    p_male = 0.5
    p_female = 1 - p_male

    cost_male = 0  # 预测男性性别的成本,设为0就是不考虑了
    cost_female = 0  # 预测女性性别的成本
    cost_false_negative = 10  # 实际为男性但预测为女性的成本
    cost_false_positive = 5  # 实际为女性但预测为男性的成本

    assert ways in [1, 2, 3], "Invalid value for 'ways'. Use 1, 2, or 3."
    assert p_male + p_female == 1., "Invalid prior probability, the sum of categories must be 1"

    # if ways == 1:
    #     assert height is not None, "If mode 1 is selected, the height parameter cannot be set to None"
    #     p_height_given_male = male_height_dist.pdf(height)
    #     p_height_given_female = female_height_dist.pdf(height)
    #
    #
    #     return 1 if p_height_given_male * p_male > p_height_given_female * p_female else 2

    if ways == 1:
        assert height is not None, "If mode 1 is selected, the height parameter cannot be set to None"
        p_height_given_male = male_height_dist.pdf(height)
        p_height_given_female = female_height_dist.pdf(height)

        risk_male = cost_male + cost_false_negative if p_height_given_male * p_male <= p_height_given_female * p_female else cost_female
        risk_female = cost_female + cost_false_positive if p_height_given_male * p_male >= p_height_given_female * p_female else cost_male

        return 1 if risk_male <= risk_female else 2

    if ways == 2:
        assert height is not None, "If mode 2 is selected, the weight parameter cannot be set to None"
        p_weight_given_male = male_weight_dist.pdf(weight)
        p_weight_given_female = female_weight_dist.pdf(weight)

        return 1 if p_weight_given_male * p_male > p_weight_given_female * p_female else 2

    if ways == 3:
        assert height is not None, "If mode 3 is selected, the height and weight parameters cannot be set to None"
        p_height_given_male = male_height_dist.pdf(height)
        p_height_given_female = female_height_dist.pdf(height)
        p_weight_given_male = male_weight_dist.pdf(weight)
        p_weight_given_female = female_weight_dist.pdf(weight)

        return 1 if p_height_given_male * p_weight_given_male * p_male > p_height_given_female * p_weight_given_female * p_female else 2

    return 3

def test(test_path,ways=3):
    test_data = np.loadtxt(test_path)
    true_gender_label=[]
    pred_gender_label=[]
    for data in test_data:
        height, weight, gender = data
        true_gender_label.append(int(gender))
        pred_gender = classify(height, weight, ways)
        pred_gender_label.append(pred_gender)
        if pred_gender == 1:
            print('Male')
        elif pred_gender == 2:
            print('Female')
        else:
            print('Unknown\t')
    return true_gender_label, pred_gender_label

def accuracy(true_labels, predicted_labels):
    assert len(true_labels) == len(predicted_labels), "Input lists must have the same length"
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total_predictions = len(true_labels)
    accuracy = correct_predictions / total_predictions
    return accuracy

if __name__=="__main__":
    male_path = r'./data\MALE.TXT'
    female_path = r'./data\FEMALE.TXT'
    test_path = r'./data/test1.txt'


    maledata_loader = Datasets(male_path, t=3)
    femaledata_loader = Datasets(female_path, t=3)


    dict = Dataloader(maledata_loader,femaledata_loader)

    male_height_dist = dict["mh"]
    female_height_dist = dict["fh"]
    male_weight_dist = dict["mw"]
    female_weight_dist = dict["fw"]

    true_gender_label, pred_gender_label = test(test_path)
    print(f"真实标签:{true_gender_label}")
    print(f"预测标签:{pred_gender_label}")
    accuracy = accuracy(true_gender_label, pred_gender_label)
    print(f"Accuracy: {round(accuracy * 100,2)}%")
