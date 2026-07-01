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

---

## 组织与社群入口

**元素碰撞 · AtomCollide-AI 智能体实验室**：面向学习者、创作者与自动化实践者，持续沉淀可复用的 AI Agent 产品、工作流与工程经验。使命：**for the learner**。

> 请选择 1 个常用社群加入，内容全域同步，无需重复加入。

### 知识库

| 知识库 | 链接 |
|---|---|
| 踩坑合集 | [进入](https://vcnvmnln7wit.feishu.cn/wiki/CjV9wG8IHiIpWikCdFEcxfErnne) |
| 商业化案例库 | [进入](https://vcnvmnln7wit.feishu.cn/wiki/LdIxwlrKGibFEVkWMocc2K9KnBh) |
| 科普专栏 | [进入](https://vcnvmnln7wit.feishu.cn/wiki/K1RPwM8zji9ZchkxlOmcivUgnJe) |
| Open Build | [进入](https://vcnvmnln7wit.feishu.cn/wiki/CThswol0PiNJJbkhgT1cZIxanLb) |
| LLM / Agent / 研究报告 | [进入](https://vcnvmnln7wit.feishu.cn/wiki/KwGQwS2TciT2EdkSBBtcYnbsnSd) |
| Skill 封装合集 | [进入](https://vcnvmnln7wit.feishu.cn/wiki/PDfpwqJZUibTyBkUa7TcZZ6Onpd) |
| 社区治理运营 | [进入](https://vcnvmnln7wit.feishu.cn/wiki/MSEGwrdnTiiF9Dk8qCVcNW6InJg) |

### 社群邀请

| 社群 | 链接 |
|---|---|
| AI 探索交流 1 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=074vd565-6084-455c-ac52-9703e89a0697) |
| AI 探索交流 2 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=60bj94f0-1a67-48a7-abbb-9172b161c2b0) |
| AI 探索交流 3 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=13do1920-db46-4444-b635-005680beaf58) |
| AI 探索交流 4 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=f17o1b86-06f6-4f10-911a-69a299a25fe3) |
| AI 探索交流 5 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=2bbh6ab6-22c2-4753-b973-74bb1a2edcc9) |
| AI 探索交流 6 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=d19r19f7-2f47-42ba-b1ec-cb0342cf2e80) |
| AI 探索交流 7 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=fe9vdacc-7316-4b4d-ae4a-fdbcf56315e6) |
| AI 探索交流 8 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=103kfae8-1fd7-424f-984f-d66c210e42d1) |
| AI 探索交流 9 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=239p3cad-2f83-4baa-a230-f40386067548) |
| AI 探索交流 10 区 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=880r7cf5-3638-45ff-afb9-7944de991872) |
| AI 探索交流 — 网文作家 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=6a3v579b-ab43-4e1a-87f9-be63bab88da7) |
| AI 探索交流群 — 音乐达人 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=76at299e-73da-4eeb-9eba-32161e98f2f8) |
| AI 探索交流群 — 微笑驿站 | [加入](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=f2av73d0-6bb4-4a9f-9095-5fbbe83e49ec) |

---

AtomCollide-智械工坊团队出品。更多产品见：[AtomCollide Product Matrix](https://503496348-ops.github.io/atomcollide-product-matrix/)。


## 示例输出

本仓库的最小可验证使用路径：

1. 阅读 README 的 Quick Start / 使用说明，完成本地安装或配置。
2. 按仓库提供的命令、脚本或入口运行一次最小任务。
3. 对照本产品定位验证输出：**此地无垠（Icilllimite）** 属于 **体育分析** 产品，目标是把输入材料转化为可检查、可复用的结果。
4. 若运行环境暂不可用，先通过 README、CHANGELOG、CI 状态和源码结构完成静态验收，再补充真实截图或录屏。

> 维护要求：后续每次发布都应把真实运行截图、CLI 输出、网页截图或 API 响应样例补充到本节，避免仓库首页只描述能力、不展示结果。

## Governance Links

- [LICENSE](LICENSE)
- [CHANGELOG](CHANGELOG.md)
- [SECURITY](SECURITY.md)
- [CONTRIBUTING](CONTRIBUTING.md)


