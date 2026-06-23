"""
Match Feature Engineering — 比赛特征工程
参考socceraction VAEP的18+特征变换器
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import math


@dataclass
class TeamForm:
    """球队近期状态"""
    team_id: str
    last_n_matches: int = 5
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    shots_per_game: float = 0.0
    possession_avg: float = 0.0
    
    @property
    def points_per_game(self) -> float:
        if self.last_n_matches == 0:
            return 0
        return (self.wins * 3 + self.draws) / self.last_n_matches
    
    @property
    def goal_difference(self) -> int:
        return self.goals_scored - self.goals_conceded
    
    @property
    def xg_difference(self) -> float:
        return self.xg_for - self.xg_against
    
    @property
    def defensive_strength(self) -> float:
        """防守强度: 低xGA = 强防守"""
        if self.last_n_matches == 0:
            return 0.5
        return 1.0 - min(self.xg_against / self.last_n_matches, 1.0)


@dataclass
class MatchFeatures:
    """比赛特征向量"""
    home_attack: float = 0.0
    home_defense: float = 0.0
    away_attack: float = 0.0
    away_defense: float = 0.0
    home_form: float = 0.0
    away_form: float = 0.0
    home_xg_diff: float = 0.0
    away_xg_diff: float = 0.0
    rest_days_home: int = 3
    rest_days_away: int = 3
    is_neutral: bool = False
    importance: float = 1.0  # 比赛重要性权重
    
    def to_vector(self) -> List[float]:
        """转换为数值向量"""
        return [
            self.home_attack, self.home_defense,
            self.away_attack, self.away_defense,
            self.home_form, self.away_form,
            self.home_xg_diff, self.away_xg_diff,
            self.rest_days_home / 7.0,  # 归一化
            self.rest_days_away / 7.0,
            1.0 if self.is_neutral else 0.0,
            self.importance,
        ]


class FeatureExtractor:
    """
    特征提取器 — 从原始比赛数据提取预测特征
    
    参考socceraction的特征变换器模式:
    - actiontype_onehot → 比赛动作类型编码
    - startpolar/endpolar → 极坐标特征
    - goalscore → 比分状态
    - team → 主客场标记
    - time_delta/space_delta → 时空差分
    """
    
    def extract_form(self, matches: List[Dict], team_id: str) -> TeamForm:
        """从近期比赛提取球队状态"""
        form = TeamForm(team_id=team_id, last_n_matches=len(matches))
        for m in matches:
            is_home = m.get('home_team') == team_id
            gf = m.get('home_goals' if is_home else 'away_goals', 0)
            ga = m.get('away_goals' if is_home else 'home_goals', 0)
            form.goals_scored += gf
            form.goals_conceded += ga
            if gf > ga:
                form.wins += 1
            elif gf == ga:
                form.draws += 1
            else:
                form.losses += 1
        return form
    
    def extract_match_features(
        self,
        home_form: TeamForm,
        away_form: TeamForm,
        rest_days_home: int = 3,
        rest_days_away: int = 3,
        is_neutral: bool = False,
        importance: float = 1.0,
    ) -> MatchFeatures:
        """提取单场比赛特征"""
        return MatchFeatures(
            home_attack=home_form.goals_scored / max(home_form.last_n_matches, 1),
            home_defense=home_form.defensive_strength,
            away_attack=away_form.goals_scored / max(away_form.last_n_matches, 1),
            away_defense=away_form.defensive_strength,
            home_form=home_form.points_per_game / 3.0,
            away_form=away_form.points_per_game / 3.0,
            home_xg_diff=home_form.xg_difference,
            away_xg_diff=away_form.xg_difference,
            rest_days_home=rest_days_home,
            rest_days_away=rest_days_away,
            is_neutral=is_neutral,
            importance=importance,
        )
