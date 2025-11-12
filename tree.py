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
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
	@@ -107,7 +107,7 @@ def choose_best_feature_split(dataset):
    base_entropy = cal_shannon_ent(dataset)
    # 3. 初始化“最大信息增益”和“最佳特征”
    best_info_gain = 0.0
    best_feature = -1
    # 4. 遍历每一个特征，计算它的信息增益
    for i in range(num_features):
        # 4.1 提取出该特征所有样本的取值列表
	@@ -153,7 +153,8 @@ def majority_cnt(class_list):
    # 2. 遍历类别列表，对每个类别进行计数
    for vote in class_list:
        # 如果该类别还未在字典中出现，先初始化计数为0
        if vote not in class_count.keys():
            class_count[vote]=0
        # 累加该类别的出现次数
        class_count[vote]+=1
    # 3. 将字典的键值对（类别, 次数）转为列表，并按次数进行降序排序
	@@ -162,7 +163,7 @@ def majority_cnt(class_list):
    # 按出现次数排序
     # 降序排列
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]

def creat_tree(dataset,labels):
    # 取出数据集中每条样本的“标签列”（通常是最后一列）
	@@ -195,6 +196,58 @@ def creat_tree(dataset,labels):
# my_data,labels=create_dataSet()
# my_tree=creat_tree(my_data,labels)

def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树进行分类预测
    
    参数：
        input_tree: 训练好的决策树
        feat_labels: 特征标签列表
        test_vec: 测试样本的特征向量
    
    返回：
        class_label: 预测的类别标签
    """
    # 获取决策树的第一个特征（根节点）
    first_str = list(input_tree.keys())[0]
    # 获取该特征对应的子树
    second_dict = input_tree[first_str]
    # 找到该特征在特征标签列表中的索引位置
    feat_index = feat_labels.index(first_str)

    # 遍历该特征的所有可能取值
    for key in second_dict.keys():
        # 如果测试样本在该特征上的值等于当前键值
        if test_vec[feat_index] == key:
            # 如果对应的值仍然是字典（表示还有子树），则递归分类
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                # 如果是叶节点，直接返回类别标签
                class_label = second_dict[key]
            return class_label

    # 如果没有找到匹配的路径，返回一个默认值（如出现次数最多的类别）
    return None

def calculate_accuracy(tree, dataset, labels):
    """
    计算训练集准确率
    """
    correct = 0
    total = len(dataset)

    for data in dataset:
        true_label = data[-1]
        features = data[:-1]
        predicted_label = classify(tree, labels, features)

        if predicted_label == true_label:
            correct += 1

    accuracy = correct / total * 100
    return accuracy


# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
	@@ -320,28 +373,75 @@ def create_plot(my_tree):
    plt.tight_layout()
    plt.show()

# # ========== 运行：建树 + 绘图 ==========
# # 示例数据集：天气与打球 (Play Tennis)
# weather_data = [
#     ['Sunny', 'Hot', 'High', False, 'No'],
#     ['Sunny', 'Hot', 'High', True, 'No'],
#     ['Overcast', 'Hot', 'High', False, 'Yes'],
#     ['Rain', 'Mild', 'High', False, 'Yes'],
#     ['Rain', 'Cool', 'Normal', False, 'Yes'],
#     ['Rain', 'Cool', 'Normal', True, 'No'],
#     ['Overcast', 'Cool', 'Normal', True, 'Yes'],
#     ['Sunny', 'Mild', 'High', False, 'No'],
#     ['Sunny', 'Cool', 'Normal', False, 'Yes'],
#     ['Rain', 'Mild', 'Normal', False, 'Yes'],
#     ['Sunny', 'Mild', 'Normal', True, 'Yes'],
#     ['Overcast', 'Mild', 'High', True, 'Yes'],
#     ['Overcast', 'Hot', 'Normal', False, 'Yes'],
#     ['Rain', 'Mild', 'High', True, 'No']
# ]

# # 特征标签
# labels = ['Outlook', 'Temperature', 'Humidity', 'Windy']

# # 生成决策树
# tree = creat_tree(weather_data, labels[:])  # 注意传入拷贝 labels[:]
# create_plot(tree)

def load_lenses_data(file_path):
    """
    读取lenses.txt文件
    """
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split('\t')
            dataset.append(items)
    return dataset

def main():
    """
    主函数：使用lenses数据集
    """
    # 加载数据
    file_path = r'C:\Users\叶雪晴\Desktop\tree\lenses.txt'
    lenses_data = load_lenses_data(file_path)

    # 特征标签
    lenses_labels = ['年龄', '屈光', '散光', '泪液分泌']

    print("数据集大小:", len(lenses_data))
    print("前5条数据:")
    for i in range(min(5, len(lenses_data))):
        print(lenses_data[i])

    # 生成决策树
    print("\n正在生成决策树...")
    tree = creat_tree(lenses_data, lenses_labels[:])
    print("决策树结构:", tree)

    # 计算训练集准确率
    accuracy = calculate_accuracy(tree, lenses_data, lenses_labels)
    print(f"\n训练集准确率: {accuracy:.2f}%")

    # 绘制决策树
    print("正在绘制决策树...")
    create_plot(tree)

if __name__ == "__main__":
    main()

# # 生成决策树
# tree = creat_tree(weather_data, labels[:])  # 注意传入拷贝 labels[:]
# create_plot(tree)
