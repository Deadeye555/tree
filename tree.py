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
    # 递归出口①：若所有样本同类，直接返回该类
    if class_list.count(class_list[0])==len(class_list):
        return class_list[0]
    # 递归出口②：若没有可用特征（只剩标签列），返回多数类
    # dataset[0] 的长度 = 特征数 + 1（标签列）
    if len(dataset[0])==1:
        return majority_cnt(class_list)
    # 选择“最优划分特征”的下标
    best_feat=choose_best_feature_split(dataset)
     # 取出该特征对应的名称（可读性用）
    best_feat_label=labels[best_feat]
    # 构建当前节点
    my_tree={best_feat_label:{}}
    del(labels[best_feat])
     # 取出该特征在所有样本上的取值列表
    feat_values=[example[best_feat] for example in dataset]
    # 去重：该特征有哪些不同的取值
    unique_vals=set(feat_values)
    # 对该特征的每个取值，分别递归构建子树
    for value in unique_vals:
        sub_labels=labels[:]   # 拷贝一份标签名列表给子递归使用
        # 把当前特征=某取值的样本切分出来
        my_tree[best_feat_label][value]=creat_tree(split_dataset(dataset,best_feat,value),sub_labels)
    return my_tree
# my_data,labels=create_dataSet()
# my_tree=creat_tree(my_data,labels)

def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树进行分类
    
    参数：
        input_tree: 训练好的决策树
        feat_labels: 特征标签列表
        test_vec: 测试样本的特征向量
    
    返回：
        分类结果
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

    # 如果测试样本的特征值不在训练树的范围内，返回None
    return None

def calculate_accuracy(tree, dataset, labels):
    """
    计算决策树在训练集上的准确率
    
    参数：
        tree: 训练好的决策树
        dataset: 训练数据集
        labels: 特征标签列表
    
    返回：
        accuracy: 准确率（0-1之间的浮点数）
    """
    correct_count = 0
    total_count = len(dataset)

    for i in range(total_count):
        test_vec = dataset[i][:-1]  # 去掉标签
        true_label = dataset[i][-1]  # 真实标签
        predicted_label = classify(tree, labels, test_vec)

        if predicted_label == true_label:
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
@@ -322,26 +374,43 @@ def create_plot(my_tree):

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
#    ['Sunny', 'Hot', 'High', False, 'No'],
#    ['Sunny', 'Hot', 'High', True, 'No'],
#    ['Overcast', 'Hot', 'High', False, 'Yes'],
#    ['Rain', 'Mild', 'High', False, 'Yes'],
#    ['Rain', 'Cool', 'Normal', False, 'Yes'],
#    ['Rain', 'Cool', 'Normal', True, 'No'],
#    ['Overcast', 'Cool', 'Normal', True, 'Yes'],
#    ['Sunny', 'Mild', 'High', False, 'No'],
#    ['Sunny', 'Cool', 'Normal', False, 'Yes'],
#    ['Rain', 'Mild', 'Normal', False, 'Yes'],
#    ['Sunny', 'Mild', 'Normal', True, 'Yes'],
#    ['Overcast', 'Mild', 'High', True, 'Yes'],
#    ['Overcast', 'Hot', 'Normal', False, 'Yes'],
#    ['Rain', 'Mild', 'High', True, 'No']
#]

# 特征标签
labels = ['Outlook', 'Temperature', 'Humidity', 'Windy']
#labels = ['Outlook', 'Temperature', 'Humidity', 'Windy']

# 生成决策树
tree = creat_tree(weather_data, labels[:])  # 注意传入拷贝 labels[:]
create_plot(tree)
#tree = creat_tree(weather_data, labels[:])  # 注意传入拷贝 labels[:]
#create_plot(tree)

lenspath = (r'D:\ppp\tree\lenses.txt')

def load_data(filepath):
    data=[]
    fr = open(filepath)
    for line in fr:
        line = line.strip().split()
        data.append(line)

    return data 
labels_lenses = ['年龄','屈光','散光','泪液分泌']
dataset = load_data(lenspath)
tree = creat_tree(dataset, labels_lenses[:])
accuracy = calculate_accuracy(tree, dataset, labels_lenses)
print(f"训练集准确率: {accuracy*100:.2f}%")
create_plot(tree)
