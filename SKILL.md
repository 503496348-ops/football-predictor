1|---
2|name: football-predictor
3|description: "ML足球比赛预测系统。RF+XGBoost+LR集成模型，27维特征，17联赛支持，胜平负+大小球预测。当需要预测足球比赛结果、分析球队数据、查看联赛排名时使用。"
4|version: 1.0.0
5|triggers:
6|  - 足球预测
7|  - 比赛预测
8|  - football predict
9|  - match prediction
10|  - 足球分析
11|  - 此地无垠
12|author: Hermes Agent
13|license: MIT
14|dependencies: [pandas, scikit-learn, xgboost, requests, numpy]
15|platforms: [linux, macos, windows]
16|metadata:
17|  hermes:
18|    tags: [football, prediction, machine-learning, sports, betting, analytics]
19|    related_skills: []
20|---

> 📖 详细技术文档见 references/ 目录
21|
22|# Football Match Predictor
23|
24|
25|## When to use
26|
27|- User asks for football/soccer match predictions
28|- User wants to analyze team form or match statistics
29|- User asks about upcoming match outcomes (胜平负, 大小球)
30|- Sports analytics, betting research, or data science demonstrations
31|
32|## Capabilities
33|
34|| Feature | Description |
35||---------|-------------|
36|| **Data Source** | football-data.co.uk (free, 20+ leagues, historical odds) |
37|| **Features** | 27-dim rolling window: wins, goals, shots, corners, form, win rate |
38|| **Models** | Voting Ensemble: Random Forest + XGBoost + Logistic Regression |
39|| **Targets** | Match Result (H/D/A) and Over/Under 2.5 goals |
40|| **Validation** | Stratified K-Fold cross-validation |
41|| **Leagues** | EPL, La Liga, Bundesliga, Serie A, Ligue 1, + 15 more |
42|
43|## Quick Start
44|
45|```bash
46|# 1. Install dependencies
47|pip install pandas scikit-learn xgboost requests numpy
48|
49|# 2. Run prediction for a league (full pipeline: download → train → predict)
50|python3 ~/.hermes/skills/football-predictor/scripts/football_predictor.py --league EPL
51|
52|# 3. Predict specific matches
53|python3 ~/.hermes/skills/football-predictor/scripts/football_predictor.py --league EPL \
54|  --predict "阿森纳 vs 切尔西" "利物浦 vs 曼城"
55|
56|# 4. List available leagues
57|
58|# 5. 预测 + 自动推送飞书卡片
59|python3 ~/.hermes/skills/football-predictor/scripts/football_predictor.py --league EPL \
60|  --predict "阿森纳 vs 切尔西" --feishu-card --feishu-chat <chat_id>
61|python3 ~/.hermes/skills/football-predictor/scripts/football_predictor.py --list-leagues
62|```
63|
64|## Supported Leagues
65|
66|| Code | League | Country |
67||------|--------|---------|
68|| EPL | Premier League | England |
69|| ELC | Championship | England |
70|| BL1 | Bundesliga | Germany |
71|| SA | Serie A | Italy |
72|| PD | La Liga | Spain |
73|| FL1 | Ligue 1 | France |
74|| PPL | Primeira Liga | Portugal |
75|| NPD | Eredivisie | Netherlands |
76|| TSL | Süper Lig | Turkey |
77|| ARG | Primera Division | Argentina |
78|| BRA | Serie A | Brazil |
79|| CHN | Super League | China |
80|| SPL | Scottish Premiership | Scotland |
81|| D1 | Superliga | Denmark |
82|| G1 | Super League | Greece |
83|| B1 | Jupiler League | Belgium |
84|| T1 | League | Japan |
85|
86|## Output Format
87|
88|Predictions are output as structured text:
89|
90|```
91|🏟️ Arsenal vs Chelsea (EPL)
92|  Home Win:  52.3%  ← predicted
93|  Draw:      24.1%
94|  Away Win:  23.6%
95|  Over 2.5:  58.7%
96|  Model Accuracy (CV): 54.2%
97|```
98|
99|## Agent Workflow
100|
101|1. Load this skill
102|2. Run `python3 <skill_path>/scripts/football_predictor.py --league <CODE>` for full analysis
103|3. Add `--predict "Team A vs Team B"` for specific match predictions
104|4. Report results to user with context (league standings, form, etc.)
105|
106|## Pitfalls
107|
108|- **First run is slow**: Downloads multi-season historical data and trains models (~2-5 min)
109|- **Data freshness**: football-data.co.uk updates weekly; predictions are only as good as the data
110|- **Not financial advice**: All predictions are for research/entertainment only
111|- **Team names must match**: Use exact team names from the league (e.g., "Man United" → "Manchester Utd"). Fuzzy matching handles partial matches but not creative abbreviations.
112|- **Odds column**: If a league lacks odds data (AvgH/AvgD/AvgA), the model skips odds features
113|- **CSV column name mismatch (critical)**: football-data.co.uk uses TWO different CSV formats:
114|  - **Main format** (mmz URLs — EPL, BL1, Serie A, La Liga, Ligue 1, etc.): `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`
115|  - **Extra format** (new URLs — CHN, ARG, BRA, JPN, DNK, GRC, etc.): `Home`, `Away`, `HG`, `AG`, `Res`
116|  - The `_standardize_columns()` function maps both to unified names. If you add a new league, verify which format its CSV uses by checking column headers.
117|- **Duplicate columns after rename**: Some leagues (e.g. Serie A) have BOTH `PSH` (native) and `PSCH` (which renames to `PSH`), creating duplicate column names. Without dedup (`df.loc[:, ~df.columns.duplicated()]`), `pd.concat` raises `InvalidIndexError`. Already handled in the script.
118|- **Season column**: Extra-format CSVs include a `Season` column natively; main-format CSVs need it added during download.
119|
120|## Provenance
121|
122|
---

## 工作流

使用此技能时，按以下步骤执行：

- [ ] 1. 确认用户需求和使用场景
- [ ] 2. 加载相关代码和配置
- [ ] 3. 执行核心功能
- [ ] 4. 验证输出结果
