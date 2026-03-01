"""
泰坦尼克号幸存者预测主程序
================================
整合预处理、模型训练、评估与可视化，输出预测结果和特征重要性图。
"""
import os
from src import preprocessing as pp
from src import model as md
from src import visualization as vis

# ==================== 配置参数 ====================
DATA_DIR = './data'               # 数据文件夹
OUTPUT_DIR = './output'            # 输出文件夹
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
RANDOM_STATE = 42
CV_SPLITS = 5

# 随机森林参数网格
PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# =================================================

def main():
    # 1. 加载数据
    train, test = pp.load_data(TRAIN_PATH, TEST_PATH)

    # 2. 预处理（缺失值填充、特征工程）
    train, test = pp.preprocess_pipeline(train, test)

    # 3. 获取特征列并准备特征矩阵
    feature_cols = pp.get_feature_columns()
    X, y, X_test, feature_names = pp.prepare_features(train, test, feature_cols)

    # 4. 模型训练与超参数调优
    best_model, grid = md.train_and_tune(X, y, PARAM_GRID,
                                         random_state=RANDOM_STATE,
                                         cv_splits=CV_SPLITS)

    # 5. 交叉验证评估（使用最佳模型）
    md.cross_validate(best_model, X, y, cv_splits=CV_SPLITS, random_state=RANDOM_STATE)

    # 6. 特征重要性可视化
    vis.plot_feature_importance(best_model, feature_names, OUTPUT_DIR)

    # 7. 生成预测并保存提交文件
    md.make_submission(best_model, X_test, test['PassengerId'], OUTPUT_DIR)

if __name__ == "__main__":
    main()