---
name: football-predictor
description: "此地无垠足球预测 — Elo+DC 模型、蒙特卡洛模拟、贝叶斯 xG"
triggers:
  - "足球预测"
  - "比赛预测"
  - "football"
  - "Elo"
  - "蒙特卡洛"
---

# Football Predictor — 此地无垠足球预测

多模型融合的足球比赛结果预测：Elo+Dixon-Coles + 蒙特卡洛模拟 + 贝叶斯 xG。

## 核心能力

| 命令 | 说明 |
|------|------|
| `football-predictor predict <home> <away>` | 预测比赛结果 |
| `football-predictor leagues` | 列出支持的联赛 |
| `football-predictor info` | 产品信息 |

Python 和 Node.js 双引擎：

```bash
# Python CLI
python3 scripts/cli.py predict brazil argentina

# Node.js 引擎（更完整）
node scripts/predict.mjs brazil argentina
```

## 架构

- `scripts/predict.mjs` — Node.js 预测主引擎
- `scripts/football_predictor.py` — Python 预测器
- `scripts/monte_carlo.py` — 蒙特卡洛模拟
- `scripts/bayesian_xg.py` — 贝叶斯 xG 模型
- `scripts/track-record.mjs` — 历史预测记录追踪

## 测试

```bash
python3 -m pytest tests/ -q
node scripts/predict.mjs --help
```
