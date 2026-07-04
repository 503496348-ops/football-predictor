#!/usr/bin/env python3
"""
Football Match Predictor — ML Ensemble (RF + XGBoost + LR)

Pipeline: Download data → Feature engineering → Train ensemble → Predict outcomes.
Data source: football-data.co.uk (free, 20+ leagues).

Usage:
    python3 football_predictor.py --league EPL
    python3 football_predictor.py --league EPL --predict "Arsenal vs Chelsea" "Liverpool vs Man City"
    python3 football_predictor.py --list-leagues
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── xG Enhancement (P2: Bayesian calibration) ──────────────────────────────
try:
    from bayesian_xg import estimate_match_xg, BayesianXGCalibrator
    HAS_XG = True
except ImportError:
    HAS_XG = False

# ─── League Config ───────────────────────────────────────────────────────────

LEAGUES = {
    "EPL":  {"name": "英超",           "country": "英格兰",  "url": "https://www.football-data.co.uk/mmz4281/{}/E0.csv",  "start": 2005},
    "ELC":  {"name": "英冠",           "country": "英格兰",  "url": "https://www.football-data.co.uk/mmz4281/{}/E1.csv",  "start": 2005},
    "BL1":  {"name": "德甲",           "country": "德国",    "url": "https://www.football-data.co.uk/mmz4281/{}/D1.csv",  "start": 2005},
    "SA":   {"name": "意甲",           "country": "意大利",  "url": "https://www.football-data.co.uk/mmz4281/{}/I1.csv",  "start": 2005},
    "PD":   {"name": "西甲",           "country": "西班牙",  "url": "https://www.football-data.co.uk/mmz4281/{}/SP1.csv", "start": 2005},
    "FL1":  {"name": "法甲",           "country": "法国",    "url": "https://www.football-data.co.uk/mmz4281/{}/F1.csv",  "start": 2005},
    "PPL":  {"name": "葡超",           "country": "葡萄牙",  "url": "https://www.football-data.co.uk/mmz4281/{}/P1.csv",  "start": 2010},
    "NPD":  {"name": "荷甲",           "country": "荷兰",    "url": "https://www.football-data.co.uk/mmz4281/{}/N1.csv",  "start": 2010},
    "TSL":  {"name": "土超",           "country": "土耳其",  "url": "https://www.football-data.co.uk/mmz4281/{}/T1.csv",  "start": 2010},
    "B1":   {"name": "比甲",           "country": "比利时",  "url": "https://www.football-data.co.uk/mmz4281/{}/B1.csv",  "start": 2005},
    "SPL":  {"name": "苏超",           "country": "苏格兰",  "url": "https://www.football-data.co.uk/mmz4281/{}/SC0.csv", "start": 2005},
    "ARG":  {"name": "阿甲",           "country": "阿根廷",  "url": "https://www.football-data.co.uk/new/ARG.csv",         "start": 2012},
    "BRA":  {"name": "巴甲",           "country": "巴西",    "url": "https://www.football-data.co.uk/new/BRA.csv",         "start": 2012},
    "CHN":  {"name": "中超",           "country": "中国",    "url": "https://www.football-data.co.uk/new/CHN.csv",         "start": 2014},
    "D1":   {"name": "丹超",           "country": "丹麦",    "url": "https://www.football-data.co.uk/new/DNK.csv",         "start": 2013},
    "G1":   {"name": "希超",           "country": "希腊",    "url": "https://www.football-data.co.uk/new/GRC.csv",         "start": 2013},
    "J1":   {"name": "J联赛",          "country": "日本",    "url": "https://www.football-data.co.uk/new/JPN.csv",         "start": 2014},
}

CACHE_DIR = Path.home() / ".hermes" / "cache" / "football"

# ─── Chinese Team Name Mapping ────────────────────────────────────────────────

TEAM_CN = {
    # === 英超 EPL ===
    "Arsenal": "阿森纳", "Aston Villa": "阿斯顿维拉", "Bournemouth": "伯恩茅斯",
    "Brentford": "布伦特福德", "Brighton": "布莱顿", "Burnley": "伯恩利",
    "Chelsea": "切尔西", "Crystal Palace": "水晶宫", "Everton": "埃弗顿",
    "Fulham": "富勒姆", "Ipswich": "伊普斯维奇", "Leeds": "利兹联",
    "Leicester": "莱斯特城", "Liverpool": "利物浦", "Luton": "卢顿",
    "Man City": "曼城", "Man United": "曼联", "Newcastle": "纽卡斯尔",
    "Nott'm Forest": "诺丁汉森林", "Sheffield United": "谢菲尔德联",
    "Southampton": "南安普顿", "Sunderland": "桑德兰", "Tottenham": "热刺",
    "West Ham": "西汉姆", "Wolves": "狼队",
    # === 德甲 BL1 ===
    "Augsburg": "奥格斯堡", "Bayern Munich": "拜仁慕尼黑", "Bielefeld": "比勒费尔德",
    "Bochum": "波鸿", "Darmstadt": "达姆施塔特", "Dortmund": "多特蒙德",
    "Ein Frankfurt": "法兰克福", "FC Koln": "科隆", "Freiburg": "弗赖堡",
    "Greuther Furth": "菲尔特", "Hamburg": "汉堡", "Heidenheim": "海登海姆",
    "Hertha": "柏林赫塔", "Hoffenheim": "霍芬海姆", "Holstein Kiel": "基尔",
    "Leverkusen": "勒沃库森", "M'gladbach": "门兴格拉德巴赫", "Mainz": "美因茨",
    "RB Leipzig": "莱比锡红牛", "Schalke 04": "沙尔克04", "St Pauli": "圣保利",
    "Stuttgart": "斯图加特", "Union Berlin": "柏林联合", "Werder Bremen": "不莱梅",
    "Wolfsburg": "沃尔夫斯堡",
    # === 意甲 SA ===
    "Atalanta": "亚特兰大", "Bologna": "博洛尼亚", "Cagliari": "卡利亚里",
    "Como": "科莫", "Cremonese": "克雷莫纳", "Empoli": "恩波利",
    "Fiorentina": "佛罗伦萨", "Frosinone": "弗罗西诺内", "Genoa": "热那亚",
    "Inter": "国际米兰", "Juventus": "尤文图斯", "Lazio": "拉齐奥",
    "Lecce": "莱切", "Milan": "AC米兰", "Monza": "蒙扎", "Napoli": "那不勒斯",
    "Parma": "帕尔马", "Pisa": "比萨", "Roma": "罗马",
    "Salernitana": "萨勒尼塔纳", "Sampdoria": "桑普多利亚", "Sassuolo": "萨索洛",
    "Spezia": "斯佩齐亚", "Torino": "都灵", "Udinese": "乌迪内斯",
    "Venezia": "威尼斯", "Verona": "维罗纳",
    # === 中超 CHN ===
    "Beijing Guoan": "北京国安", "Beijing Renhe": "北京人和", "Cangzhou": "沧州雄狮",
    "Changchun Yatai": "长春亚泰", "Chengdu Rongcheng": "成都蓉城",
    "Chongqing Liangjiang Athletic": "重庆两江竞技", "Chongqing Tonglianglong": "重庆铜梁龙",
    "Dalian Pro": "大连人", "Dalian Yifang F.C.": "大连一方", "Dalian Yingbo": "大连英博",
    "Guangzhou City": "广州城", "Guangzhou Evergrande": "广州恒大",
    "Guangzhou FC": "广州队", "Guangzhou R&F": "广州富力", "Guizhou Zhicheng": "贵州智诚",
    "Hangzhou Greentown": "杭州绿城", "Hebei": "河北队", "Henan Songshan Longmen": "河南嵩山龙门",
    "Jiangsu Suning": "江苏苏宁", "Liaoning": "辽宁队", "Liaoning Tieren": "辽宁铁人",
    "Meizhou Hakka": "梅州客家", "Nantong Zhiyun": "南通支云", "Qingdao FC": "青岛队",
    "Qingdao Hainiu": "青岛海牛", "Qingdao West Coast": "青岛西海岸",
    "Shandong Luneng": "山东鲁能", "Shandong Taishan": "山东泰山",
    "Shanghai Port": "上海海港", "Shanghai SIPG": "上海上港",
    "Shanghai Shenhua": "上海申花", "Shanghai Shenxin": "上海申鑫",
    "Shenzhen": "深圳队", "Shenzhen Xinpengcheng": "深圳新鹏城",
    "Shijiazhuang": "石家庄永昌", "Tianjin Jinmen Tiger": "天津津门虎",
    "Tianjin Quanjian": "天津权健", "Tianjin Tianhai": "天津天海",
    "Wuhan FC": "武汉队", "Wuhan Three Towns": "武汉三镇", "Yanbian": "延边富德",
    "Yunnan Yukun": "云南玉昆", "Zhejiang Greentown": "浙江绿城",
    "Zhejiang Professional": "浙江队", "Zhejiang Yiteng": "浙江毅腾",
    # === 法甲 FL1 ===
    "Paris Saint Germain": "巴黎圣日耳曼", "Marseille": "马赛", "Lyon": "里昂",
    "Monaco": "摩纳哥", "Lille": "里尔", "Nice": "尼斯", "Rennes": "雷恩",
    "Lens": "朗斯", "Strasbourg": "斯特拉斯堡", "Nantes": "南特",
    # === 西甲 PD ===
    "Real Madrid": "皇家马德里", "Barcelona": "巴塞罗那", "Atletico Madrid": "马德里竞技",
    "Sevilla": "塞维利亚", "Real Sociedad": "皇家社会", "Villarreal": "比利亚雷亚尔",
    "Athletic Club": "毕尔巴鄂竞技", "Valencia": "瓦伦西亚", "Getafe": "赫塔费",
    "Celta": "塞尔塔", "Osasuna": "奥萨苏纳", "Girona": "赫罗纳",
    "Betis": "贝蒂斯", "Mallorca": "马洛卡", "Las Palmas": "拉斯帕尔马斯",
    # === 葡超 PPL ===
    "Benfica": "本菲卡", "Porto": "波尔图", "Sporting CP": "葡萄牙体育",
    "Braga": "布拉加",
    # === 英冠 ELC ===
    "Leeds": "利兹联", "Burnley": "伯恩利", "Sheffield United": "谢菲尔德联",
}

# Reverse mapping: Chinese → English
TEAM_EN = {v: k for k, v in TEAM_CN.items()}


# ─── Data Download ───────────────────────────────────────────────────────────

def download_league_data(league_code: str, seasons: int = 5) -> pd.DataFrame:
    """Download historical match data from football-data.co.uk."""
    cfg = LEAGUES[league_code]
    current_year = datetime.now().year
    frames = []

    # For "new" format URLs (ARG, BRA, CHN, etc.), download once
    if "/new/" in cfg["url"]:
        url = cfg["url"]
        try:
            df = pd.read_csv(url, on_bad_lines="skip")
            df = _standardize_columns(df)
            if "Date" in df.columns and len(df) > 0:
                frames.append(df)
        except Exception as e:
            print(f"  ⚠ Download failed for {url}: {e}", file=sys.stderr)
    else:
        # For "mmz" format, iterate over season ranges
        for year_offset in range(seasons):
            y1 = current_year - year_offset - 1
            y2 = current_year - year_offset
            season_str = f"{str(y1)[-2:]}{str(y2)[-2:]}"
            url = cfg["url"].format(season_str)
            try:
                df = pd.read_csv(url, on_bad_lines="skip")
                df = _standardize_columns(df)
                if "Date" in df.columns and len(df) > 0:
                    df["Season"] = f"{y1}/{y2}"
                    frames.append(df)
            except Exception:
                continue

    if not frames:
        raise ValueError(f"No data downloaded for {league_code}. Check league code and internet connection.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    combined = combined.sort_values("Date").reset_index(drop=True)
    return combined


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column names across different data formats."""
    rename_map = {
        # Main format (mmz URLs)
        "HomeTeam": "HomeTeam", "AwayTeam": "AwayTeam",
        "FTHG": "FTHG", "FTAG": "FTAG", "FTR": "FTR",
        "HS": "HS", "AS": "AS", "HST": "HST", "AST": "AST",
        "HC": "HC", "AC": "AC",
        "BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA",
        "AvgH": "AvgH", "AvgD": "AvgD", "AvgA": "AvgA",
        # Extra format (new URLs: CHN, ARG, BRA, etc.)
        "Home": "HomeTeam", "Away": "AwayTeam",
        "HG": "FTHG", "AG": "FTAG", "Res": "FTR",
        "AvgCH": "AvgH", "AvgCD": "AvgD", "AvgCA": "AvgA",
        "PSCH": "PSH", "PSCD": "PSD", "PSCA": "PSA",
    }
    # Only rename columns that exist
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)

    # Deduplicate: if renaming created duplicate column names, drop extras
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Parse date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    return df


# ─── Feature Engineering (27-dim rolling window) ─────────────────────────────

def compute_team_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute 27-dimensional rolling window team statistics.
    Based on rolling window statistics engine.

    Features per match (home team stats vs away team stats):
    - Wins, Losses (home/away/combined)
    - Goals Forward, Goals Against, Goal Difference
    - Win Rate, Loss Rate
    - Shots on Target, Corners (if available)
    """
    # Build per-team match history
    teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

    # Create match rows for each team
    records = []
    for _, row in df.iterrows():
        # Home team perspective
        records.append({
            "team": row["HomeTeam"], "opponent": row["AwayTeam"],
            "is_home": 1, "goals_for": row["FTHG"], "goals_against": row["FTAG"],
            "result": row["FTR"], "date": row["Date"],
            "shots_on_target": row.get("HST", np.nan),
            "corners": row.get("HC", np.nan),
            "avg_odds_h": row.get("AvgH", np.nan),
            "avg_odds_d": row.get("AvgD", np.nan),
            "avg_odds_a": row.get("AvgA", np.nan),
        })
        # Away team perspective
        records.append({
            "team": row["AwayTeam"], "opponent": row["HomeTeam"],
            "is_home": 0, "goals_for": row["FTAG"], "goals_against": row["FTHG"],
            "result": "A" if row["FTR"] == "H" else ("H" if row["FTR"] == "A" else "D"),
            "date": row["Date"],
            "shots_on_target": row.get("AST", np.nan),
            "corners": row.get("AC", np.nan),
            "avg_odds_h": row.get("AvgA", np.nan),
            "avg_odds_d": row.get("AvgD", np.nan),
            "avg_odds_a": row.get("AvgH", np.nan),
        })

    # xG estimation from shots (if HS/AS/HST/AST available)
    if "HS" in df.columns and "HST" in df.columns:
        SOT_CONV, SHOT_CONV = 0.32, 0.03
        for _, row in df.iterrows():
            hs = float(row.get("HS", 0) or 0)
            hst = float(row.get("HST", 0) or 0)
            as_ = float(row.get("AS", 0) or 0)
            ast = float(row.get("AST", 0) or 0)
            xg_home = hst * SOT_CONV + max(0, hs - hst) * SHOT_CONV
            xg_away = ast * SOT_CONV + max(0, as_ - ast) * SHOT_CONV
            # Find matching records in the records list and add xg
            for r in records:
                if r["date"] == row["Date"]:
                    if r["is_home"] == 1 and r["team"] == row["HomeTeam"]:
                        r["xg_for"] = xg_home
                        r["xg_against"] = xg_away
                    elif r["is_home"] == 0 and r["team"] == row["AwayTeam"]:
                        r["xg_for"] = xg_away
                        r["xg_against"] = xg_home

        team_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

        # Rolling xG features
        if "xg_for" in team_df.columns:
            team_df["xg_diff"] = team_df["xg_for"] - team_df["xg_against"]
            for col, prefix in [("xg_for", "xg_for"), ("xg_against", "xg_against"),
                                ("xg_diff", "xg_diff")]:
                h_mask = team_df["is_home"] == 1
                team_df.loc[h_mask, f"h_{prefix}"] = (
                    team_df.loc[h_mask].groupby("team")[col].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
                    )
                )
                team_df.loc[~h_mask, f"a_{prefix}"] = (
                    team_df.loc[~h_mask].groupby("team")[col].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
                    )
                )
            # xG efficiency: actual goals / xG (rolling)
            team_df["xg_eff"] = team_df["goals_for"] / team_df["xg_for"].clip(lower=0.01)
            h_mask = team_df["is_home"] == 1
            team_df.loc[h_mask, "h_xg_eff"] = (
                team_df.loc[h_mask].groupby("team")["xg_eff"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=3).mean()
                )
            )
            team_df.loc[~h_mask, "a_xg_eff"] = (
                team_df.loc[~h_mask].groupby("team")["xg_eff"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=3).mean()
                )
            )
    else:
        team_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # Compute rolling features per team
    team_df["win"] = (team_df["result"] == "H").astype(int)
    team_df["draw"] = (team_df["result"] == "D").astype(int)
    team_df["loss"] = (team_df["result"] == "A").astype(int)
    team_df["gd"] = team_df["goals_for"] - team_df["goals_against"]
    team_df["over25"] = ((team_df["goals_for"] + team_df["goals_against"]) > 2.5).astype(int)

    # Rolling window features
    roll_cols = {
        "win": "wins", "loss": "losses", "draw": "draws",
        "goals_for": "gf", "goals_against": "ga", "gd": "gd",
        "over25": "over25_count",
    }

    for col, prefix in roll_cols.items():
        # Home team rolling
        h_mask = team_df["is_home"] == 1
        team_df.loc[h_mask, f"h_{prefix}"] = (
            team_df.loc[h_mask].groupby("team")[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=3).sum()
            )
        )
        # Away team rolling
        team_df.loc[~h_mask, f"a_{prefix}"] = (
            team_df.loc[~h_mask].groupby("team")[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=3).sum()
            )
        )

    # Shots and corners (if available)
    for col in ["shots_on_target", "corners"]:
        if team_df[col].notna().any():
            h_mask = team_df["is_home"] == 1
            team_df.loc[h_mask, f"h_{col}"] = (
                team_df.loc[h_mask].groupby("team")[col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=3).mean()
                )
            )
            team_df.loc[~h_mask, f"a_{col}"] = (
                team_df.loc[~h_mask].groupby("team")[col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=3).mean()
                )
            )

    # Win rate
    team_df["h_win_rate"] = team_df["h_wins"] / window
    team_df["a_win_rate"] = team_df["a_wins"] / window
    team_df["h_loss_rate"] = team_df["h_losses"] / window
    team_df["a_loss_rate"] = team_df["a_losses"] / window

    return team_df


def build_match_features(df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """Merge home and away team stats into match-level features."""
    # Home team stats
    home_cols = ["team", "date", "h_wins", "h_losses", "h_draws", "h_gf", "h_ga", "h_gd",
                 "h_win_rate", "h_loss_rate", "h_over25_count"]
    for opt in ["h_shots_on_target", "h_corners", "h_xg_for", "h_xg_against", "h_xg_diff", "h_xg_eff"]:
        if opt in team_df.columns:
            home_cols.append(opt)
    home_stats = team_df[team_df["is_home"] == 1][home_cols].rename(columns={"team": "HomeTeam", "date": "Date"})

    # Away team stats
    away_cols = ["team", "date", "a_wins", "a_losses", "a_draws", "a_gf", "a_ga", "a_gd",
                 "a_win_rate", "a_loss_rate", "a_over25_count"]
    for opt in ["a_shots_on_target", "a_corners", "a_xg_for", "a_xg_against", "a_xg_diff", "a_xg_eff"]:
        if opt in team_df.columns:
            away_cols.append(opt)
    away_stats = team_df[team_df["is_home"] == 0][away_cols].rename(columns={"team": "AwayTeam", "date": "Date"})

    # Merge with original match data
    merged = df.merge(home_stats, on=["HomeTeam", "Date"], how="left")
    merged = merged.merge(away_stats, on=["AwayTeam", "Date"], how="left")

    # Add odds if available
    if "AvgH" in merged.columns:
        merged["odds_home"] = merged["AvgH"].fillna(0)
        merged["odds_draw"] = merged["AvgD"].fillna(0)
        merged["odds_away"] = merged["AvgA"].fillna(0)

    return merged


# ─── Model Training ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "h_wins", "h_losses", "h_draws", "h_gf", "h_ga", "h_gd",
    "h_win_rate", "h_loss_rate", "h_over25_count",
    "a_wins", "a_losses", "a_draws", "a_gf", "a_ga", "a_gd",
    "a_win_rate", "a_loss_rate", "a_over25_count",
]

OPTIONAL_FEATURES = [
    "h_shots_on_target", "h_corners", "a_shots_on_target", "a_corners",
    "odds_home", "odds_draw", "odds_away",
    # xG features (P2: Bayesian xG enhancement)
    "h_xg_for", "h_xg_against", "h_xg_diff", "h_xg_eff",
    "a_xg_for", "a_xg_against", "a_xg_diff", "a_xg_eff",
]

TARGET_MAP = {"H": 0, "D": 1, "A": 2}


def train_model(match_df: pd.DataFrame, target: str = "result"):
    """Train a Voting Ensemble (RF + XGBoost + LR)."""
    # Prepare features
    available_features = [c for c in FEATURE_COLS if c in match_df.columns]
    available_features += [c for c in OPTIONAL_FEATURES if c in match_df.columns]

    train_df = match_df.dropna(subset=available_features + ["FTR"])
    if len(train_df) < 100:
        raise ValueError(f"Not enough training data ({len(train_df)} matches). Need at least 100.")

    X = train_df[available_features].values

    if target == "result":
        y = train_df["FTR"].map(TARGET_MAP).values
        labels = ["Home Win", "Draw", "Away Win"]
    elif target == "over_under":
        y = ((train_df["FTHG"] + train_df["FTAG"]) > 2.5).astype(int).values
        labels = ["Under 2.5", "Over 2.5"]
    else:
        raise ValueError(f"Unknown target: {target}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build ensemble
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                            use_label_encoder=False, eval_metric="mlogloss",
                            verbosity=0, random_state=42)
    except ImportError:
        xgb = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    if target == "result":
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb), ("lr", lr)],
            voting="soft"
        )
    else:
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb), ("lr", lr)],
            voting="soft"
        )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in cv.split(X_scaled, y):
        ensemble.fit(X_scaled[train_idx], y[train_idx])
        score = accuracy_score(y[val_idx], ensemble.predict(X_scaled[val_idx]))
        cv_scores.append(score)

    # Final training on all data
    ensemble.fit(X_scaled, y)

    return {
        "model": ensemble,
        "scaler": scaler,
        "features": available_features,
        "cv_accuracy": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "labels": labels,
        "target": target,
        "n_train": len(train_df),
    }


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_match(model_info: dict, home_team: str, away_team: str,
                  home_stats: dict, away_stats: dict) -> dict:
    """Predict a single match outcome with Monte Carlo uncertainty layer."""
    from monte_carlo import monte_carlo_simulate, analytic_score_grid

    features = model_info["features"]
    scaler = model_info["scaler"]
    model = model_info["model"]
    labels = model_info["labels"]

    # Build feature vector
    x = []
    for f in features:
        if f.startswith("h_"):
            x.append(home_stats.get(f, 0))
        elif f.startswith("a_"):
            x.append(away_stats.get(f, 0))
        elif f.startswith("odds_"):
            x.append(0)  # No odds for upcoming matches
        else:
            x.append(0)

    x = np.array(x).reshape(1, -1)
    x_scaled = scaler.transform(x)

    # Get ML probabilities
    probs = model.predict_proba(x_scaled)[0]
    pred_idx = np.argmax(probs)

    # Estimate expected goals for Monte Carlo
    # Use xG features if available, otherwise derive from team stats
    lambda_home = home_stats.get("h_xg_for", home_stats.get("h_gf", 1.3))
    lambda_away = away_stats.get("a_xg_for", away_stats.get("a_gf", 1.1))

    # Ensure positive
    lambda_home = max(0.3, float(lambda_home))
    lambda_away = max(0.3, float(lambda_away))

    # Monte Carlo simulation (10k sims, Poisson + lambda uncertainty)
    mc = monte_carlo_simulate(lambda_home, lambda_away, rho=-0.13, n_sim=10000, lambda_uncertainty=0.15)
    analytic = analytic_score_grid(lambda_home, lambda_away, rho=-0.13)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": labels[pred_idx],
        "probabilities": {label: float(prob) for label, prob in zip(labels, probs)},
        "confidence": float(probs[pred_idx]),
        "expectedGoals": {"home": round(lambda_home, 2), "away": round(lambda_away, 2)},
        "monteCarlo": mc,
        "analyticGrid": analytic,
    }


def get_team_recent_stats(team_df: pd.DataFrame, team_name: str, is_home: bool) -> dict:
    """Get the most recent rolling stats for a team."""
    mask = (team_df["team"] == team_name) & (team_df["is_home"] == (1 if is_home else 0))
    team_data = team_df[mask].sort_values("date", ascending=False)

    if team_data.empty:
        return {}

    row = team_data.iloc[0]
    stats = {}
    prefix = "h" if is_home else "a"
    for col in team_data.columns:
        if col.startswith(f"{prefix}_") and pd.notna(row[col]):
            stats[col] = float(row[col])
    return stats


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(league_code: str, predict_matches: list = None,
                 cache: bool = True, seasons: int = 5,
                 feishu_card: bool = False, feishu_chat: str = None):
    """Full pipeline: download → engineer features → train → predict."""
    print(f"\n⚽ 此地无垠 · 足球预测 — {LEAGUES[league_code]['name']} ({league_code})")
    print("=" * 60)

    # Step 1: Download data
    print(f"\n📥 加载 {seasons} 赛季数据...")
    cache_path = CACHE_DIR / f"{league_code}.pkl"

    if cache and cache_path.exists():
        df = pd.read_pickle(cache_path)
        print(f"  ✓ 缓存加载 {len(df)} 场比赛")
    else:
        df = download_league_data(league_code, seasons=seasons)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if cache:
            df.to_pickle(cache_path)
        print(f"  ✓ 下载完成 {len(df)} 场比赛")

    # Step 2: Feature engineering
    print("\n🔧 计算27维滚动特征 + xG贝叶斯校准 (窗口=10)...")
    team_df = compute_team_stats(df, window=10)
    match_df = build_match_features(df, team_df)
    print(f"  ✓ Features computed for {len(match_df)} matches")

    # Step 3: Train model
    print("\n🧠 训练 RF + XGBoost + LR 集成模型 (5折交叉验证)...")
    result_info = train_model(match_df, target="result")
    print(f"  ✓ 训练完成 {result_info['n_train']} 场比赛")
    print(f"  ✓ 准确率: {result_info['cv_accuracy']:.1%} ± {result_info['cv_std']:.1%}")

    print("\n🧠 训练大小球模型...")
    ou_info = train_model(match_df, target="over_under")
    print(f"  ✓ 大小球准确率: {ou_info['cv_accuracy']:.1%} ± {ou_info['cv_std']:.1%}")

    # Step 4: Predict matches
    all_predictions = []
    if predict_matches:
        print(f"\n🏟️ 预测 {len(predict_matches)} 场比赛...")
        print("-" * 50)

        for match_str in predict_matches:
            # Parse "Team A vs Team B"
            if " vs " in match_str.lower():
                parts = match_str.lower().split(" vs ")
            elif " v " in match_str.lower():
                parts = match_str.lower().split(" v ")
            else:
                print(f"  ⚠ 无法解析: {match_str} (请用 '队伍A vs 队伍B' 格式)")
                continue

            home_input = parts[0].strip()
            away_input = parts[1].strip()

            # Support Chinese input
            if home_input in TEAM_EN:
                home_input = TEAM_EN[home_input]
            else:
                home_input = home_input.title()
            if away_input in TEAM_EN:
                away_input = TEAM_EN[away_input]
            else:
                away_input = away_input.title()

            # Find matching team names (fuzzy)
            all_teams = set(df["HomeTeam"].unique())
            home_match = _find_team(all_teams, home_input)
            away_match = _find_team(all_teams, away_input)

            if not home_match or not away_match:
                missing = home_input if not home_match else away_input
                sample = [_cn(t) for t in sorted(all_teams)[:15]]
                print(f"  ⚠ 未找到球队: {_cn(missing)}")
                print(f"    可用: {', '.join(sample)}...")
                continue

            # Get recent stats
            h_stats = get_team_recent_stats(team_df, home_match, is_home=True)
            a_stats = get_team_recent_stats(team_df, away_match, is_home=False)

            # Result prediction
            result_pred = predict_match(result_info, home_match, away_match, h_stats, a_stats)
            # Over/Under prediction
            ou_pred = predict_match(ou_info, home_match, away_match, h_stats, a_stats)

            all_predictions.append({
                "home": home_match, "away": away_match,
                "home_cn": _cn(home_match), "away_cn": _cn(away_match),
                "result_pred": result_pred["prediction"],
                "result_probs": result_pred["probabilities"],
                "ou_pred": ou_pred["prediction"],
                "ou_probs": ou_pred["probabilities"],
            })

            print(f"\n  ⚽ {_cn(home_match)} vs {_cn(away_match)}")
            p = result_pred["probabilities"]
            arrow = " ←" if result_pred["prediction"] == "Home Win" else ""
            print(f"    主胜:  {p.get('Home Win', 0):5.1%}{arrow}")
            arrow = " ←" if result_pred["prediction"] == "Draw" else ""
            print(f"    平局:  {p.get('Draw', 0):5.1%}{arrow}")
            arrow = " ←" if result_pred["prediction"] == "Away Win" else ""
            print(f"    客胜:  {p.get('Away Win', 0):5.1%}{arrow}")

            ou_p = ou_pred["probabilities"]
            ou_arrow = " ←" if ou_pred["prediction"] == "Over 2.5" else ""
            print(f"    大2.5: {ou_p.get('Over 2.5', 0):5.1%}{ou_arrow}")

    # Show top teams by form
    print("\n📊 近期主场战绩 Top 10")
    print("-" * 50)
    latest_home = team_df[team_df["is_home"] == 1].sort_values("date", ascending=False)
    latest_home = latest_home.drop_duplicates(subset=["team"], keep="first")

    top_teams_data = []
    if "h_win_rate" in latest_home.columns:
        top = latest_home.nlargest(10, "h_win_rate")[["team", "h_win_rate", "h_gf", "h_ga", "h_gd"]]
        for _, row in top.iterrows():
            top_teams_data.append({
                "team": row["team"], "win_rate": row["h_win_rate"],
                "gf": row["h_gf"], "ga": row["h_ga"], "gd": row["h_gd"]
            })
            print(f"  {_cn(row['team']):12s}  胜率: {row['h_win_rate']:.0%}  进球:{row['h_gf']:.0f}  失球:{row['h_ga']:.0f}  净胜:{row['h_gd']:+.0f}")

    print(f"\n✅ Pipeline complete. CV Accuracy: {result_info['cv_accuracy']:.1%}")

    # Build and send Feishu card
    if feishu_card and all_predictions:
        card = build_feishu_card(league_code, all_predictions, top_teams_data,
                                 result_info["cv_accuracy"], result_info["n_train"])
        send_feishu_card(card, chat_id=feishu_chat)

    return result_info, ou_info


def _cn(name: str) -> str:
    """Return Chinese name for a team, falling back to original."""
    return TEAM_CN.get(name, name)


def _find_team(all_teams: set, name: str) -> str:
    """Fuzzy match team name (supports Chinese input)."""
    # Exact match (English)
    if name in all_teams:
        return name
    # Chinese → English lookup
    if name in TEAM_EN:
        en = TEAM_EN[name]
        if en in all_teams:
            return en
    # Case-insensitive
    for t in all_teams:
        if t.lower() == name.lower():
            return t
    # Chinese partial match
    for cn, en in TEAM_EN.items():
        if name in cn or cn in name:
            if en in all_teams:
                return en
    # English partial match
    for t in all_teams:
        if name.lower() in t.lower() or t.lower() in name.lower():
            return t
    return None


def list_leagues():
    """Print available leagues."""
    print("\n⚽ 可用联赛")
    print("=" * 50)
    for code, cfg in sorted(LEAGUES.items()):
        print(f"  {code:5s}  {cfg['name']:8s}  ({cfg['country']})")


# ─── Feishu Card Builder ─────────────────────────────────────────────────────

def build_feishu_card(league_code: str, predictions: list, top_teams: list,
                      cv_accuracy: float, n_train: int) -> dict:
    """Build a Feishu interactive card from prediction results."""
    cfg = LEAGUES[league_code]

    # Color scheme per match
    bg_colors = ["blue-50", "violet-50", "purple-50", "green-50", "orange-50"]
    text_colors = ["blue", "violet", "purple", "green", "orange"]

    match_elements = []
    for i, pred in enumerate(predictions):
        tc = text_colors[i % len(text_colors)]
        result_p = pred["result_probs"]
        ou_p = pred["ou_probs"]

        # Highlight the predicted outcome
        r_pred = pred["result_pred"]
        lines = []
        for label, key in [("主胜", "Home Win"), ("平局", "Draw"), ("客胜", "Away Win")]:
            val = result_p.get(key, 0)
            if key == r_pred:
                lines.append(f"{label} <font color='{tc}'>`{val:.1%}`</font> ←")
            else:
                lines.append(f"{label} `{val:.1%}`")
        ou_val = ou_p.get("Over 2.5", 0)
        ou_mark = " ←" if pred["ou_pred"] == "Over 2.5" else ""
        lines.append(f"大2.5 `{ou_val:.1%}`{ou_mark}")

        match_elements.append({
            "tag": "markdown",
            "content": f"**🏟️ {pred['home_cn']} vs {pred['away_cn']}**\n{' | '.join(lines)}"
        })

    # Top teams table rows
    table_rows = []
    for t in top_teams[:5]:
        table_rows.append({
            "team": _cn(t["team"]),
            "win_rate": f"{t['win_rate']:.0%}",
            "goals": f"{t['gf']:.0f}",
            "losses": f"{t['ga']:.0f}",
            "net": f"{t['gd']:+.0f}"
        })

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "text_tag_list": [{"color": "blue", "tag": "text_tag", "text": {"content": f"{cfg['name']} {league_code}", "tag": "plain_text"}}],
            "title": {"content": "⚽ 此地无垠 · 足球预测", "tag": "plain_text"}
        },
        "elements": [
            {
                "tag": "markdown",
                "content": f"模型准确率 **{cv_accuracy:.1%}** | 训练数据 **{n_train}场**"
            },
            {"tag": "hr"},
            *match_elements,
            {"tag": "hr"},
            {
                "tag": "markdown",
                "content": "**📊 近期主场战绩 Top 5**"
            },
            {
                "tag": "table",
                "header_style": {"background_style": "none"},
                "columns": [
                    {"data_type": "text", "display_name": "球队", "name": "team", "width": "auto"},
                    {"data_type": "text", "display_name": "胜率", "name": "win_rate", "width": "auto"},
                    {"data_type": "text", "display_name": "进球", "name": "goals", "width": "auto"},
                    {"data_type": "text", "display_name": "失球", "name": "losses", "width": "auto"},
                    {"data_type": "text", "display_name": "净胜", "name": "net", "width": "auto"}
                ],
                "rows": table_rows
            },
            {
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": "⚠️ 仅作技术研究与娱乐参考，不构成投注建议 | 数据源: football-data.co.uk"}]
            }
        ]
    }
    return card


def send_feishu_card(card: dict, chat_id: str = None):
    """Send a Feishu interactive card via lark-cli."""
    import subprocess

    profile = os.environ.get("HERMES_LARK_CLI_PROFILE", "default")
    target_chat = chat_id or os.environ.get("HERMES_SESSION_CHAT_ID")

    if not target_chat:
        # Save card to file for manual sending
        out_path = CACHE_DIR / "last_card.json"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=2)
        print(f"  ℹ️ 未指定 chat_id，卡片已保存到 {out_path}")
        return False

    cmd = [
        "lark-cli", "--profile", profile,
        "im", "+messages-send",
        "--chat-id", target_chat,
        "--msg-type", "interactive",
        "--as", "bot",
        "--content", json.dumps(card, ensure_ascii=False)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            resp = json.loads(result.stdout)
            if resp.get("ok"):
                print(f"  ✅ 卡片已推送到 {target_chat}")
                return True
            else:
                print(f"  ⚠️ 推送失败: {resp}")
        else:
            print(f"  ⚠️ lark-cli 错误: {result.stderr[:200]}")
    except Exception as e:
        print(f"  ⚠️ 推送异常: {e}")

    # Fallback: save to file
    out_path = CACHE_DIR / "last_card.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"  📄 卡片 JSON 已保存到 {out_path}")
    return False


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Football Match Predictor (ML Ensemble)")
    parser.add_argument("--league", "-l", type=str, help="League code (e.g., EPL, BL1, SA)")
    parser.add_argument("--predict", "-p", nargs="+", help='Matches to predict (e.g., "Arsenal vs Chelsea")')
    parser.add_argument("--list-leagues", action="store_true", help="List available leagues")
    parser.add_argument("--seasons", "-s", type=int, default=5, help="Number of seasons to download (default: 5)")
    parser.add_argument("--no-cache", action="store_true", help="Disable data caching")
    parser.add_argument("--feishu-card", action="store_true", help="Generate and send Feishu card")
    parser.add_argument("--feishu-chat", type=str, help="Feishu chat_id for card delivery")

    args = parser.parse_args()

    if args.list_leagues:
        list_leagues()
        return

    if not args.league:
        parser.error("--league is required (use --list-leagues to see options)")

    league = args.league.upper()
    if league not in LEAGUES:
        parser.error(f"Unknown league: {league}. Use --list-leagues to see options.")

    run_pipeline(
        league_code=league,
        predict_matches=args.predict,
        cache=not args.no_cache,
        seasons=args.seasons,
        feishu_card=args.feishu_card,
        feishu_chat=args.feishu_chat,
    )


if __name__ == "__main__":
    main()
