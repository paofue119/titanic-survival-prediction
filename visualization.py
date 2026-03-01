"""
可视化模块：包含特征重要性绘图等。
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_feature_importance(model, feature_names, output_dir, save=True):
    """绘制特征重要性柱状图并保存"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("特征重要性 (Feature Importances)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(path, dpi=120)
        print("特征重要性图已保存至:", path)
    plt.show()