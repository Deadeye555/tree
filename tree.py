import operator
from math import log
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==========================================
# 1. 核心算法部分 (熵、划分、建树)
# ==========================================

def cal_shannon_ent(dataset):
    """ 计算香农熵 """
    num_entries = len(dataset)
    labels_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
        labels_counts[current_label] += 1
    
    shannon_ent = 0.0
    for key in labels_counts:
        prob = float(labels_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def split_dataset(dataset, axis, value):
    """ 按照给定特征划分数据集 """
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            # 去掉 axis 这一列
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset

def choose_best_feature_split(dataset):
    """ 选择最好的特征进行划分 """
    num_features = len(dataset[0]) - 1
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)
        new_entropy = 0.0
        
        # 计算条件熵
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * cal_shannon_ent(sub_dataset)
        
        info_gain = base_entropy - new_entropy
        
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
            
    # 如果没有特征能带来增益（例如所有特征都一样但标签不同），返回0或其他默认值
    if best_feature == -1:
        return 0
    return best_feature

def majority_cnt(class_list):
    """ 多数表决，返回出现次数最多的类别 """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def create_tree(dataset, labels):
    """ 递归构建决策树 """
    class_list = [example[-1] for example in dataset]
    
    # 递归出口 1: 类别完全相同
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    
    # 递归出口 2: 遍历完所有特征，返回多数表决结果
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    
    best_feat = choose_best_feature_split(dataset)
    best_feat_label = labels[best_feat]
    
    my_tree = {best_feat_label: {}}
    del(labels[best_feat]) # 注意：这会修改传入的列表，建议调用时传入副本
    
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    
    for value in unique_vals:
        sub_labels = labels[:] # 复制标签列表，防止递归污染
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)
        
    return my_tree

def classify(input_tree, feature_labels, test_vec):
    """ 使用决策树进行分类 """
    first_str = next(iter(input_tree))
    child_dict = input_tree[first_str]
    
    try:
        feat_index = feature_labels.index(first_str)
    except ValueError:
        return "Unknown Feature"

    key = test_vec[feat_index]
    
    # 如果遇到未知的特征值（训练集中没出现过），返回无法判断
    if key not in child_dict:
        # 简单策略：返回当前节点下出现最多的叶子，或者直接返回 unknown
        return "Unknown Value"
        
    feat_value = child_dict[key]
    
    if isinstance(feat_value, dict):
        return classify(feat_value, feature_labels, test_vec)
    else:
        return feat_value

# ==========================================
# 2. 可视化部分 (Matplotlib)
# ==========================================

# 定义节点样式
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def get_num_leafs(my_tree):
    """ 获取叶子节点数量（确定图的宽度） """
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    """ 获取树的深度（确定图的高度） """
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    """ 绘制节点 """
    ax.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center", bbox=node_type, arrowprops=arrow_args)

def plot_mid_text(ax, center_pt, parent_pt, txt_string):
    """ 在连线中间标注特征取值 """
    x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
    y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)

def plot_tree_recursive(ax, my_tree, parent_pt, node_txt, total_w, total_d, x_off_y):
    """ 递归绘制树 """
    num_leafs = get_num_leafs(my_tree)
    first_str = next(iter(my_tree))
    # 计算当前节点的中心位置
    center_pt = (x_off_y['x'] + (1.0 + float(num_leafs)) / 2.0 / total_w, x_off_y['y'])
    
    # 绘制连线上的文字
    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)
    
    # 绘制决策节点
    plot_node(ax, first_str, center_pt, parent_pt, decision_node)
    
    second_dict = my_tree[first_str]
    # y 坐标偏移，进入下一层
    x_off_y['y'] = x_off_y['y'] - 1.0 / total_d
    
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree_recursive(ax, second_dict[key], center_pt, str(key), total_w, total_d, x_off_y)
        else:
            # 绘制叶子节点
            x_off_y['x'] = x_off_y['x'] + 1.0 / total_w
            plot_node(ax, str(second_dict[key]), (x_off_y['x'], x_off_y['y']), center_pt, leaf_node)
            plot_mid_text(ax, (x_off_y['x'], x_off_y['y']), center_pt, str(key))
            
    # 递归结束，y 坐标回退
    x_off_y['y'] = x_off_y['y'] + 1.0 / total_d

def create_plot(my_tree):
    """ 主绘图函数 """
    fig = plt.figure(1, facecolor='white', figsize=(10, 6)) # 增大画布尺寸
    fig.clf()
    ax = plt.subplot(111, frameon=False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    
    # x_off: x轴的起始偏移量
    # y_off: y轴的起始偏移量 (根节点在最上面 1.0)
    x_off_y = {'x': -0.5 / total_w, 'y': 1.0}
    
    plot_tree_recursive(ax, my_tree, (0.5, 1.0), '', total_w, total_d, x_off_y)
    plt.show()

# ==========================================
# 3. 主流程与测试
# ==========================================

def load_lenses(path):
    """ 读取数据文件 """
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    data.append(parts)
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {path}")
        return []

if __name__ == "__main__":
    # 1. 准备数据
    # 请修改这里的路径为你本地的真实路径
    file_path = r'C:\Users\E507\Desktop\tree\lenses.txt' 
    lenses_data = load_lenses(file_path)

    if lenses_data:
        # 定义特征名称
        lenses_labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
        
        # 2. 构建树
        # 注意：create_tree 会修改 labels 列表，所以这里传入 labels[:] 的副本
        print("正在构建决策树...")
        lenses_tree = create_tree(lenses_data, lenses_labels[:])
        print("决策树结构:", lenses_tree)
        
        # 3. 绘制树
        print("正在绘制决策树...")
        create_plot(lenses_tree)
        
        # 4. 计算准确率 (Training Accuracy)
        # 注意：这里需要重新定义一遍 labels，因为之前的 labels 已经被修改可能为空
        X_labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
        
        correct = 0
        total = len(lenses_data)
        
        for row in lenses_data:
            x = row[:-1]
            y_true = row[-1]
            y_pred = classify(lenses_tree, X_labels, x)
            if y_pred == y_true:
                correct += 1
        
        accuracy = correct / total
        print("=" * 30)
        print(f"训练集样本数: {total}")
        print(f"预测正确数: {correct}")
        print(f"训练集准确率: {accuracy:.2%}")
        print("=" * 30)
