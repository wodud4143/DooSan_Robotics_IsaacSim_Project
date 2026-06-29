"""Shared runtime handles initialized by bootstrap.py.

Isaac Sim objects are created in a strict order, so modules read these handles
after bootstrap() has opened the stage.
"""

CLI_ARGS = None
WORLD_USD_PATH = None
SORTER_ROS_TOPIC = None
SPAWN_INTERVAL_SEC = None
SPAWN_ON_START = False

simulation_app = None
usd_context = None
stage = None
physx_query = None

ROS_NODE = None
L_ROS_NODE = None
CONVEYOR_ROS_NODE = None
