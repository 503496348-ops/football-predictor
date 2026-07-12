from pipeline import DataPipeline


def test_statsbomb_adapter_normalizes_shot_coordinates_and_goal_result():
    events = [{
        "id": "evt-1", "index": 7, "period": 1, "minute": 12, "second": 30,
        "team": {"name": "Home"}, "player": {"id": 9, "name": "Striker"},
        "type": {"name": "Shot"}, "shot": {"outcome": {"name": "Goal"}, "end_location": [120, 40]},
        "location": [102, 32],
    }]
    match = DataPipeline().from_statsbomb_events("m1", "Home", "Away", events)
    assert match.home_goals == 1
    assert match.events[0].event_type == "shot"
    assert match.events[0].result == "goal"
    assert match.events[0].start_x > 80
