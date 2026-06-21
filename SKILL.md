---
name: football-predictor
description: "Football match outcome predictor using ML ensemble. Downloads data from football-data.co.uk, computes 27-dim rolling team features, trains RF+XGBoost+LR ensemble, predicts match results (H/D/A) and Over/Under 2.5 goals."
version: 1.0.0
author: Hermes Agent
license: MIT
dependencies: [pandas, scikit-learn, xgboost, requests, numpy]
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [football, prediction, machine-learning, sports, betting, analytics]
    related_skills: []
---

# Football Match Predictor


## When to use

- User asks for football/soccer match predictions
- User wants to analyze team form or match statistics
- User asks about upcoming match outcomes (胜平负, 大小球)
- Sports analytics, betting research, or data science demonstrations

## Capabilities

| Feature | Description |
|---------|-------------|
| **Data Source** | football-data.co.uk (free, 20+ leagues, historical odds) |
| **Features** | 27-dim rolling window: wins, goals, shots, corners, form, win rate |
| **Models** | Voting Ensemble: Random Forest + XGBoost + Logistic Regression |
| **Targets** | Match Result (H/D/A) and Over/Under 2.5 goals |
| **Validation** | Stratified K-Fold cross-validation |
| **Leagues** | EPL, La Liga, Bundesliga, Serie A, Ligue 1, + 15 more |

## Quick Start

```bash
# 1. Install dependencies
pip install pandas scikit-learn xgboost requests numpy

# 2. Run prediction for a league (full pipeline: download → train → predict)
python3 ~/.hermes/skills/football-predictor/scripts/football_predictor.py --league EPL

# 3. Predict specific matches
python3 ~/.hermes/skills/football-predictor/scripts/football_predictor.py --league EPL \
  --predict "阿森纳 vs 切尔西" "利物浦 vs 曼城"

# 4. List available leagues

# 5. 预测 + 自动推送飞书卡片
python3 ~/.hermes/skills/football-predictor/scripts/football_predictor.py --league EPL \
  --predict "阿森纳 vs 切尔西" --feishu-card --feishu-chat <chat_id>
python3 ~/.hermes/skills/football-predictor/scripts/football_predictor.py --list-leagues
```

## Supported Leagues

| Code | League | Country |
|------|--------|---------|
| EPL | Premier League | England |
| ELC | Championship | England |
| BL1 | Bundesliga | Germany |
| SA | Serie A | Italy |
| PD | La Liga | Spain |
| FL1 | Ligue 1 | France |
| PPL | Primeira Liga | Portugal |
| NPD | Eredivisie | Netherlands |
| TSL | Süper Lig | Turkey |
| ARG | Primera Division | Argentina |
| BRA | Serie A | Brazil |
| CHN | Super League | China |
| SPL | Scottish Premiership | Scotland |
| D1 | Superliga | Denmark |
| G1 | Super League | Greece |
| B1 | Jupiler League | Belgium |
| T1 | League | Japan |

## Output Format

Predictions are output as structured text:

```
🏟️ Arsenal vs Chelsea (EPL)
  Home Win:  52.3%  ← predicted
  Draw:      24.1%
  Away Win:  23.6%
  Over 2.5:  58.7%
  Model Accuracy (CV): 54.2%
```

## Agent Workflow

1. Load this skill
2. Run `python3 <skill_path>/scripts/football_predictor.py --league <CODE>` for full analysis
3. Add `--predict "Team A vs Team B"` for specific match predictions
4. Report results to user with context (league standings, form, etc.)

## Pitfalls

- **First run is slow**: Downloads multi-season historical data and trains models (~2-5 min)
- **Data freshness**: football-data.co.uk updates weekly; predictions are only as good as the data
- **Not financial advice**: All predictions are for research/entertainment only
- **Team names must match**: Use exact team names from the league (e.g., "Man United" → "Manchester Utd"). Fuzzy matching handles partial matches but not creative abbreviations.
- **Odds column**: If a league lacks odds data (AvgH/AvgD/AvgA), the model skips odds features
- **CSV column name mismatch (critical)**: football-data.co.uk uses TWO different CSV formats:
  - **Main format** (mmz URLs — EPL, BL1, Serie A, La Liga, Ligue 1, etc.): `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`
  - **Extra format** (new URLs — CHN, ARG, BRA, JPN, DNK, GRC, etc.): `Home`, `Away`, `HG`, `AG`, `Res`
  - The `_standardize_columns()` function maps both to unified names. If you add a new league, verify which format its CSV uses by checking column headers.
- **Duplicate columns after rename**: Some leagues (e.g. Serie A) have BOTH `PSH` (native) and `PSCH` (which renames to `PSH`), creating duplicate column names. Without dedup (`df.loc[:, ~df.columns.duplicated()]`), `pd.concat` raises `InvalidIndexError`. Already handled in the script.
- **Season column**: Extra-format CSVs include a `Season` column natively; main-format CSVs need it added during download.

## Provenance

