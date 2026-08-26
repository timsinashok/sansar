from sansar.core.types import Action, CarState
from sansar.envs.driving.physics import ClassicEngine
from sansar.utils.config import load_config


def make_engine(*overrides):
    return ClassicEngine(load_config(list(overrides)).env)


def test_straight_driving_advances_distance():
    eng = make_engine()
    s = eng.reset()
    for _ in range(100):
        s = eng.step(s, Action.NONE)
    assert s.distance > 0
    assert s.x == 0.0
    assert not s.collided


def test_steering_moves_laterally():
    eng = make_engine()
    s = eng.reset()
    for _ in range(50):
        s = eng.step(s, Action.RIGHT)
    assert s.x > 0
    assert s.heading > 0


def test_collision_clamps_to_edge_and_slows():
    eng = make_engine()
    s = eng.reset()
    for _ in range(2000):
        s = eng.step(s, Action.RIGHT)
    limit = eng.road.half_width - eng.car_half_width
    assert s.collided
    assert abs(s.x) <= limit + 1e-9
    assert s.speed >= eng.collision_min_speed


def test_state_array_roundtrip():
    s = CarState(x=1.5, heading=-0.2, speed=9.0, distance=123.4, collided=True)
    assert CarState.from_array(s.to_array()) == CarState.from_array(
        CarState.from_array(s.to_array()).to_array()
    )


def test_heading_self_centers_after_release():
    eng = make_engine()
    s = eng.reset()
    for _ in range(20):
        s = eng.step(s, Action.RIGHT)
    assert s.heading > 0
    for _ in range(100):
        s = eng.step(s, Action.NONE)
    assert s.heading == 0.0


def test_obstacle_layout_is_deterministic_and_on_road():
    eng = make_engine()
    obs_a = eng.obstacles.in_range(0, 1000)
    obs_b = eng.obstacles.in_range(0, 1000)
    assert obs_a == obs_b
    assert len(obs_a) > 0
    for ob in obs_a:
        edge = eng.road.half_width - ob.half_width
        assert abs(ob.x - eng.road.center_at(ob.distance)) <= edge


def test_hitting_obstacle_slows_car():
    eng = make_engine()
    ob = eng.obstacles.in_range(0, 1000)[0]
    s = CarState(
        x=ob.x,
        heading=0.0,
        speed=eng.max_speed,
        distance=ob.distance - ob.half_length,
        collided=False,
    )
    s = eng.step(s, Action.NONE)
    assert s.collided
    assert s.speed < eng.max_speed


def test_obstacles_can_be_disabled():
    eng = make_engine("env.obstacles.enabled=false")
    assert eng.obstacles.in_range(0, 1000) == []


def test_curved_road_center_moves():
    eng = make_engine("env.road.kind=curved")
    assert eng.road.center_at(20.0) != eng.road.center_at(0.0)
