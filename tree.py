from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib
def cal_shannon_ent(dataset):
    """
    计算熵
    """
    # 1. 计算数据集中样本的总数
    num_entries = len(dataset)
    # 2. 创建一个字典，用于统计每个类别标签出现的次数
    labels_counts = {}
    # 3. 遍历数据集中的每条记录
    for feat_vec in dataset:
        # feat_vec[-1] 表示每条样本的最后一个元素 类别标签
        current_label = feat_vec[-1]
        # 如果该标签是第一次出现，则在字典中初始化为 0
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
        # 累加该标签出现的次数
        labels_counts[current_label] += 1
        #print("类别统计：", labels_counts)
    # 4. 计算香农熵
    shannon_ent = 0.0
    # 遍历字典中的每个类别及其计数
    for key in labels_counts:
        # 计算该类别的概率
        prob = float(labels_counts[key])/num_entries
        # 根据香农熵公式累加：
        shannon_ent -= prob*log(prob, 2)
    # 5. 返回计算得到的熵值
    return shannon_ent
def create_dataSet():
    """
    熵接近 1，说明“yes”和“no”两个类别的比例比较接近，数据集的不确定性较高。
    熵接近 0,类别越集中，数据集越“纯”或“确定性越强”
    """
    dataset = [[1, 1, 'yes'],
               [1.1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no suerfacing', 'flippers']
    return dataset, labels
dataset, labels = create_dataSet()
print(cal_shannon_ent(dataset))
def split_dataset(dataset, axis, value):
    """
    按照指定特征(axis)的某个取值(value)划分数据集。
    会选出所有该特征等于 value 的样本，
    并且返回时会去掉这一列特征。
    参数：
        dataset: 原始数据集（二维列表，每一行是一个样本，每一列是一个特征，最后一列通常是标签）
        axis: 要划分的特征列索引（例如 0 表示第 1 个特征）
        value: 特征的目标取值（例如 'sunny'）
    返回：
        ret_dataset: 划分后的子数据集（不包含 axis 那一列）
    """
    ret_dataset = []  # 用于存放划分后的子数据集
    # 遍历原始数据集的每一条样本
    for feat_vec in dataset:
        # 如果这一条样本在 axis 特征上的值等于给定的 value
        if feat_vec[axis] == value:
            # 构建一个“去掉该特征”的新样本
            reduced_feat_vec = feat_vec[:axis]    # 取前面部分
            reduced_feat_vec.extend(feat_vec[axis+1:])  # 取后面部分拼接起来
            # 把这个新样本加入到子数据集中
            ret_dataset.append(reduced_feat_vec)
      # 返回划分后的数据集
    return ret_dataset
# 示例数据集：最后一列是标签
dataset_test = [
    [1, 'sunny', 'yes'],
    [1, 'rainy', 'no'],
    [0, 'sunny', 'yes']
]
# 按第0列的值为1来划分
result = split_dataset(dataset_test, 0, 1)
print(result)
def choose_best_feature_split(dataset):
    """
    选择信息增益最大的特征索引，作为本轮划分的最优特征。
    参数：
        dataset: 数据集（二维列表，每行一条样本，最后一列是标签）
    返回：
        best_feature: 最优特征的索引位置
    """
    # 1. 计算特征总数（最后一列是标签，不算特征）
    num_features = len(dataset[0])-1
    # 2. 计算原始数据集的熵（未划分前的不确定性）
    base_entropy = cal_shannon_ent(dataset)
    # 3. 初始化“最大信息增益”和“最佳特征”
    best_info_gain = 0.0
    best_feature = 1
    # 4. 遍历每一个特征，计算它的信息增益
    for i in range(num_features):
        # 4.1 提取出该特征所有样本的取值列表
        feat_list = [example[i] for example in dataset]
        #这是一个列表推导式的写法
        #等价于:
        #feat_list = []
        #for example in dataset:
        #    feat_list.append(example[i])
        # 4.2 获取该特征的所有唯一取值,转换为set集合，自动去重
        unique_val = set(feat_list)
        # 4.3 计算该特征划分后的“加权平均熵”
        new_entropy = 0.0
        for value in unique_val:
            # 按照该特征的某个取值划分数据集
            sub_dataset = split_dataset(dataset, i, value)
             # 计算该子集占整个数据集的比例
            prob = len(sub_dataset)/float(len(dataset))
            # 累加加权熵（概率 * 子集熵）
            new_entropy += prob*cal_shannon_ent(sub_dataset)
        # 4.4 计算该特征的信息增益
        info_gain = base_entropy-new_entropy
        # 4.5 如果当前特征信息增益更大，就更新最优特征
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    # 5. 返回信息增益最大的特征索引
    return best_feature
#print(choose_best_feature_split(loan_data))
def majority_cnt(class_list):
    """
    功能：统计 class_list 中各类别出现的次数，并按出现次数从多到少排序返回。
    参数：
        class_list: 列表，例如 ['yes', 'no', 'yes', 'yes', 'no']
    返回：
        一个按类别出现次数从多到少排列的列表，例如：
        [('yes', 3), ('no', 2)]
    """
     # 1. 定义一个空字典，用于存放每个类别及其计数
    class_count={}
    # 2. 遍历类别列表，对每个类别进行计数
    for vote in class_list:
        # 如果该类别还未在字典中出现，先初始化计数为0
        if vote not in class_count.keys():class_count[vote]=0
        # 累加该类别的出现次数
        class_count[vote]+=1
    # 3. 将字典的键值对（类别, 次数）转为列表，并按次数进行降序排序
    # operator.itemgetter(1) 表示按照元组中第2个元素（计数）排序
    # dict.items() => [('yes',3), ('no',2)]
    # 按出现次数排序
     # 降序排列
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count


def creat_tree(dataset,labels):
    # 取出数据集中每条样本的“标签列”（通常是最后一列）
    class_list=[example[-1] for example in dataset]
@@ -196,6 +197,90 @@ def creat_tree(dataset,labels):
# my_tree=creat_tree(my_data,labels)



def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树进行分类预测
    
    参数:
        input_tree: 训练好的决策树
        feat_labels: 特征标签列表
        test_vec: 测试样本特征向量
    返回:
        预测的类别标签
    """
    first_str = next(iter(input_tree))
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)

    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
            return class_label


def detailed_accuracy(tree, dataset, labels):
    """
    计算详细准确率统计
    """
    correct_count = 0
    total_count = len(dataset)
    confusion_info = {}

    for data in dataset:
        features = data[:-1]
        true_label = data[-1]
        predicted_label = classify(tree, labels, features)

        # 统计正确率
        if predicted_label == true_label:
            correct_count += 1

        # 统计混淆信息
        if true_label not in confusion_info:
            confusion_info[true_label] = {'correct': 0, 'total': 0, 'errors': {}}

        confusion_info[true_label]['total'] += 1

        if predicted_label == true_label:
            confusion_info[true_label]['correct'] += 1
        else:
            if predicted_label not in confusion_info[true_label]['errors']:
                confusion_info[true_label]['errors'][predicted_label] = 0
            confusion_info[true_label]['errors'][predicted_label] += 1

    # 计算总体准确率
    overall_accuracy = correct_count / total_count

    # 打印详细结果
    print("=" * 50)
    print("训练集准确率分析报告")
    print("=" * 50)
    print(f"总体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"正确分类: {correct_count}/{total_count}")
    print(f"错误分类: {total_count - correct_count}/{total_count}")
    print()

    # 打印每个类别的准确率
    print("各类别准确率:")
    for true_label, info in confusion_info.items():
        class_accuracy = info['correct'] / info['total']
        print(f"  {true_label}: {info['correct']}/{info['total']} ({class_accuracy:.2%})")

        # 打印错误分类详情
        if info['errors']:
            print(f"    错误分类为: {info['errors']}")

    return overall_accuracy






# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
@@ -322,26 +407,63 @@ def create_plot(my_tree):

# ========== 运行：建树 + 绘图 ==========
# 示例数据集：天气与打球 (Play Tennis)
weather_data = [
    ['Sunny', 'Hot', 'High', False, 'No'],
    ['Sunny', 'Hot', 'High', True, 'No'],
    ['Overcast', 'Hot', 'High', False, 'Yes'],
    ['Rain', 'Mild', 'High', False, 'Yes'],
    ['Rain', 'Cool', 'Normal', False, 'Yes'],
    ['Rain', 'Cool', 'Normal', True, 'No'],
    ['Overcast', 'Cool', 'Normal', True, 'Yes'],
    ['Sunny', 'Mild', 'High', False, 'No'],
    ['Sunny', 'Cool', 'Normal', False, 'Yes'],
    ['Rain', 'Mild', 'Normal', False, 'Yes'],
    ['Sunny', 'Mild', 'Normal', True, 'Yes'],
    ['Overcast', 'Mild', 'High', True, 'Yes'],
    ['Overcast', 'Hot', 'Normal', False, 'Yes'],
    ['Rain', 'Mild', 'High', True, 'No']
]
#weather_data = [
    #['Sunny', 'Hot', 'High', True, 'No'],
    #['Overcast', 'Hot', 'High', False, 'Yes'],
    #['Rain', 'Mild', 'High', False, 'Yes'],
    #['Rain', 'Cool', 'Normal', False, 'Yes'],
    #['Overcast', 'Cool', 'Normal', True, 'Yes'],
    #['Sunny', 'Mild', 'High', False, 'No'],
    #['Sunny', 'Cool', 'Normal', False, 'Yes'],
    #['Sunny', 'Mild', 'Normal', True, 'Yes'],
    #['Overcast', 'Mild', 'High', True, 'Yes'],
    #['Overcast', 'Hot', 'Normal', False, 'Yes'],
    #['Rain', 'Mild', 'High', True, 'No']
#]

# 特征标签
labels = ['Outlook', 'Temperature', 'Humidity', 'Windy']
#labels = ['Outlook', 'Temperature', 'Humidity', 'Windy']

# 生成决策树
tree = creat_tree(weather_data, labels[:])  # 注意传入拷贝 labels[:]
create_plot(tree)
#tree = creat_tree(weather_data, labels[:])  # 注意传入拷贝 labels[:]
#create_plot(tree)

# lenses_data = (r'E:\物联网\machine learning\tree\lenses.txt')

# def load_data(filepath):
#     data = []
#     fr = open(filepath)
#     for line in fr:
#         line = line.strip().split('\t') 
#         data.append(line)
#     return data
# labels_lenses = ['年龄', '屈光','散光','泪液分泌']
# dataset = load_data(lenses_data)
# tree = creat_tree(dataset,labels_lenses[:])
# create_plot(tree)

# 主程序
if __name__ == "__main__":
    lenses_data = r'E:\物联网\machine learning\tree\lenses.txt'

    def load_data(filepath):
        data = []
        fr = open(filepath, 'r', encoding='utf-8')
        for line in fr:
            line = line.strip().split('\t') 
            data.append(line)
        fr.close()
        return data

    labels_lenses = ['年龄', '屈光','散光','泪液分泌']
    dataset = load_data(lenses_data)

    # 构建决策树
    tree = creat_tree(dataset, labels_lenses[:])

    # 计算准确率
    accuracy = detailed_accuracy(tree, dataset, labels_lenses)

    # 可视化决策树
    create_plot(tree)
#
