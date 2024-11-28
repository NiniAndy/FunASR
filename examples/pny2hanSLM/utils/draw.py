import matplotlib.pyplot as plt


def draw_sores(text,correct_test, scores):
    characters = []
    for i in range(len(text)):
        if text[i] == correct_test[i]:
            characters.append(text[i])
        else:
            characters.append(text[i]+"/"+correct_test[i])
    # characters = [i for i in text]
    x = [i for i in range(len(text))]

    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



    # 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(x, scores, marker='o')  # 使用'o'标记每个数据点
    plt.xticks(x, characters)  # 将x轴标记设置为字符

    # 添加图表标题和坐标轴标签
    plt.title("Character Scores")
    plt.xlabel("Character")
    plt.ylabel("Score")

    # # 旋转x轴上的标签，以便它们更容易阅读
    # plt.xticks(rotation=-45)

    # 显示图表
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()