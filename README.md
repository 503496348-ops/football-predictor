# 此地无垠 · Football Predictor

足球比赛结果预测工具 — 基于机器学习集成模型的比赛结果预测。

## 概述

此地无垠使用多种ML模型集成预测足球比赛结果：
- **数据源**: football-data.co.uk 历史比赛数据
- **特征工程**: 27维滚动特征（进攻/防守/状态/xG）
- **模型集成**: Random Forest + XGBoost + Logistic Regression
- **预测输出**: 胜/平/负概率 + 大小球2.5预测

## 快速开始

```bash
# 安装
pip install -r requirements.txt

# 下载数据
python predictor.py --download

# 预测比赛
python predictor.py --home "Manchester City" --away "Liverpool"
```

## 使用场景

- **赛前分析**: 预测比赛胜平负概率和大小球
- **投注辅助**: 基于概率的期望价值计算
- **赛季模拟**: 模拟完整赛季排名
- **球队评估**: 基于xG的球队攻防能力评分

## 技术架构

- **xG模型**: 基于极坐标特征的Expected Goals估算
- **特征工程**: 近期状态、主客场、休息天数、比赛重要性
- **数据管线**: 采集→清洗→标准化→特征提取→预测
- **球场网格**: 16×12网格的Expected Threat (xT)模型

## API

```python
from features import FeatureExtractor, TeamForm
from xg_model import XGModel, ShotEvent

# 预测射门xG
xg = XGModel()
shot = ShotEvent(x=80, y=34, body_part='foot', situation='open_play')
print(f"xG: {xg.predict(shot):.3f}")

# 提取比赛特征
extractor = FeatureExtractor()
features = extractor.extract_match_features(home_form, away_form)
```

## License

MIT
