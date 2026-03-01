"""
模型训练模块：包含随机森林训练、超参数调优、交叉验证、预测提交。
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import pandas as pd
import os

def train_and_tune(X, y, param_grid, random_state=42, cv_splits=5):
    """使用网格搜索训练随机森林，返回最佳模型"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X, y)
    print("\n网格搜索完成。")
    print("最佳参数:", grid.best_params_)
    print("最佳交叉验证准确率: {:.4f}".format(grid.best_score_))
    return grid.best_estimator_, grid

def cross_validate(model, X, y, cv_splits=5, random_state=42):
    """对模型进行交叉验证并打印每折准确率"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print("\n{}-折交叉验证准确率:".format(cv_splits))
    for i, score in enumerate(scores, 1):
        print("  第{}折: {:.4f}".format(i, score))
    print("  平均准确率: {:.4f} (+/- {:.4f})".format(scores.mean(), scores.std() * 2))
    return scores

def make_submission(model, X_test, test_ids, output_dir):
    """生成预测并保存提交文件"""
    predictions = model.predict(X_test)
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(path, index=False)
    print("预测结果已保存至:", path)
    return submission