"""
Data Pipeline — 比赛数据采集与标准化管线
统一数据格式（参考SPADL标准化思路）
"""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime


@dataclass
class Player:
    player_id: str
    name: str
    position: str = ""
    team_id: str = ""


@dataclass
class MatchEvent:
    """统一比赛事件表示（参考SPADL Schema）"""
    match_id: str
    event_id: int
    period: int           # 1=上半场, 2=下半场, 3/4=加时
    time_seconds: float
    team_id: str
    player_id: str
    event_type: str       # pass/shot/tackle/interception/dribble/clearance...
    result: str           # success/fail/goal/blocked/missed
    start_x: float = 0.0
    start_y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    body_part: str = "foot"
    
    # 派生字段
    @property
    def time_minute(self) -> int:
        return int(self.time_seconds / 60)
    
    @property
    def distance_covered(self) -> float:
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        return (dx**2 + dy**2) ** 0.5


@dataclass
class MatchData:
    """比赛完整数据"""
    match_id: str
    home_team: str
    away_team: str
    competition: str = ""
    date: str = ""
    home_goals: int = 0
    away_goals: int = 0
    events: List[MatchEvent] = field(default_factory=list)
    
    @property
    def total_events(self) -> int:
        return len(self.events)
    
    @property
    def total_shots(self) -> int:
        return sum(1 for e in self.events if e.event_type == 'shot')
    
    @property
    def possession_home(self) -> float:
        if not self.events:
            return 0.5
        home_count = sum(1 for e in self.events if e.team_id == self.home_team)
        return home_count / len(self.events)


class DataPipeline:
    """
    数据管线 — 采集→清洗→标准化→存储
    
    支持数据源:
    - JSON文件导入
    - CSV批量导入
    - API实时采集（预留接口）
    """
    
    REQUIRED_FIELDS = ['match_id', 'home_team', 'away_team']
    
    def validate(self, data: dict) -> List[str]:
        """数据质量验证"""
        errors = []
        for f in self.REQUIRED_FIELDS:
            if f not in data:
                errors.append(f"缺少必填字段: {f}")
        if 'events' in data:
            for i, evt in enumerate(data['events']):
                if 'event_type' not in evt:
                    errors.append(f"事件#{i}: 缺少event_type")
        return errors
    
    def normalize_coordinates(self, events: List[MatchEvent]) -> List[MatchEvent]:
        """坐标标准化到105×68m"""
        for evt in events:
            # 如果坐标在0-1范围，转换为米
            if 0 <= evt.start_x <= 1 and 0 <= evt.start_y <= 1:
                evt.start_x *= 105.0
                evt.start_y *= 68.0
                evt.end_x *= 105.0
                evt.end_y *= 68.0
        return events
    
    def to_json(self, match: MatchData) -> str:
        """序列化为JSON"""
        return json.dumps(asdict(match), ensure_ascii=False, indent=2)
    
    def from_json(self, json_str: str) -> MatchData:
        """从JSON反序列化"""
        data = json.loads(json_str)
        events = [MatchEvent(**e) for e in data.pop('events', [])]
        return MatchData(events=events, **data)
