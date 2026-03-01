# 🚢 Titanic Survival Prediction (泰坦尼克号生存预测)

## 📖 项目简介
本项目基于机器学习算法预测泰坦尼克号乘客的生存情况。通过深入的探索性数据分析 (EDA)、特征工程以及模型调优，最终在 Kaggle 竞赛中取得了 **0.78229** 的准确率。

## 🛠 技术栈
- **语言**: Python 3.8+
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **机器学习**: Scikit-Learn (RandomForest, GridSearchCV)
- **工程化**: 模块化代码结构，Git 版本控制

## 📊 核心工作
1. **数据清洗**: 处理缺失值 (Age, Embarked, Fare)，采用众数/中位数填充。
2. **特征工程**: 
   - 构建 `FamilySize` (家庭规模) 和 `IsChild` (是否儿童) 特征。
   - 从 `Name` 中提取 `Title` (称谓) 并合并稀有类别。
3. **模型优化**: 
   - 使用 **Stratified K-Fold** 交叉验证防止过拟合。
   - 通过 **GridSearchCV** 对随机森林超参数进行调优。
4. **结果**: 本地 CV 分数稳定在 0.82+，Kaggle 公共榜单得分 0.78229。

## 🚀 如何运行
1. 克隆仓库：`git clone https://github.com/yourname/titanic-survival.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 放入数据：将 `train.csv` 和 `test.csv` 放入 `data/` 目录
4. 运行脚本：`python main.py`

## 📈 未来改进
- 尝试 XGBoost/LightGBM 集成学习。
- 引入更多特征交互 (如 Pclass * Age)。
