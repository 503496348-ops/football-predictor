"""
Expected Goals (xG) Model — 基于socceraction的极坐标特征工程
参考: ML-KULeuven/socceraction VAEP框架
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShotEvent:
    """射门事件 — 统一表示"""
    x: float           # 射门位置x (0-105m, 标准化球场)
    y: float           # 射门位置y (0-68m)
    body_part: str     # foot/foot_left/foot_right/head/other
    situation: str     # open_play/penalty/freekick/corner/set_piece
    is_first_time: bool = False
    is_volley: bool = False
    is_header: bool = False
    distance_to_goal: Optional[float] = None
    angle_to_goal: Optional[float] = None

    def __post_init__(self):
        goal_x, goal_y = 105.0, 34.0  # 球门中心
        dx = goal_x - self.x
        dy = goal_y - self.y
        self.distance_to_goal = math.sqrt(dx**2 + dy**2)
        self.angle_to_goal = math.degrees(math.atan2(abs(dy), max(dx, 0.01)))


class XGModel:
    """
    Expected Goals模型 — 基于极坐标特征的xG估算
    
    特征工程（参考socceraction xT框架）:
    - 距离球门距离 (polar distance)
    - 射门角度 (polar angle)
    - 身体部位 (body part encoding)
    - 比赛情境 (situation encoding)
    - 技术动作标记 (first_time/volley/header)
    
    模型: 逻辑回归 + 特征交叉项（可升级为XGBoost）
    """
    
    # 基于历史数据的特征权重（逻辑回归系数近似）
    WEIGHTS = {
        'intercept': -1.5,
        'distance': -0.08,      # 距离越远xG越低
        'angle': 0.02,          # 角度越大(越正对球门)xG越高
        'angle_sq': -0.0001,    # 角度二次项
        'head': -0.3,           # 头球降低xG
        'first_time': -0.15,    # 第一时间射门降低xG
        'volley': -0.2,         # 凌空降低xG
        'penalty': 2.5,         # 点球大幅提高
        'freekick': -0.5,       # 任意球降低
        'corner': -0.8,         # 角球降低
    }
    
    def predict(self, shot: ShotEvent) -> float:
        """预测射门的xG值 (0-1)"""
        if shot.situation == 'penalty':
            return 0.76  # 点球历史平均
        
        import math as m
        z = self.WEIGHTS['intercept']
        z += self.WEIGHTS['distance'] * shot.distance_to_goal
        z += self.WEIGHTS['angle'] * shot.angle_to_goal
        z += self.WEIGHTS['angle_sq'] * (shot.angle_to_goal ** 2)
        
        if shot.is_header:
            z += self.WEIGHTS['head']
        if shot.is_first_time:
            z += self.WEIGHTS['first_time']
        if shot.is_volley:
            z += self.WEIGHTS['volley']
        if shot.situation == 'freekick':
            z += self.WEIGHTS['freekick']
        if shot.situation == 'corner':
            z += self.WEIGHTS['corner']
        
        return 1.0 / (1.0 + m.exp(-z))
    
    def predict_batch(self, shots: list) -> list:
        """批量预测xG"""
        return [self.predict(s) for s in shots]


class PitchGrid:
    """
    球场网格模型 — 参考socceraction xT (Expected Threat)框架
    将球场划分为16×12网格，计算每个格子的得分/推进威胁值
    """
    
    GRID_X = 16  # 进攻方向网格数
    GRID_Y = 12  # 横向网格数
    FIELD_X = 105.0  # 球场长度(m)
    FIELD_Y = 68.0   # 球场宽度(m)
    
    def __init__(self):
        self.cell_x = self.FIELD_X / self.GRID_X
        self.cell_y = self.FIELD_Y / self.GRID_Y
        self.threat_matrix = [[0.0] * self.GRID_Y for _ in range(self.GRID_X)]
    
    def get_cell(self, x: float, y: float) -> tuple:
        """坐标→网格索引"""
        cx = min(int(x / self.cell_x), self.GRID_X - 1)
        cy = min(int(y / self.cell_y), self.GRID_Y - 1)
        return (max(0, cx), max(0, cy))
    
    def compute_threat(self, shot_data: list) -> list:
        """
        基于历史射门数据计算各网格威胁值
        shot_data: [{'x': float, 'y': float, 'xg': float}, ...]
        """
        counts = [[0] * self.GRID_Y for _ in range(self.GRID_X)]
        goals = [[0.0] * self.GRID_Y for _ in range(self.GRID_X)]
        
        for shot in shot_data:
            cx, cy = self.get_cell(shot['x'], shot['y'])
            counts[cx][cy] += 1
            goals[cx][cy] += shot.get('xg', 0)
        
        for i in range(self.GRID_X):
            for j in range(self.GRID_Y):
                if counts[i][j] > 0:
                    self.threat_matrix[i][j] = goals[i][j] / counts[i][j]
        
        return self.threat_matrix
