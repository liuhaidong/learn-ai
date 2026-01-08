"""
Configuration for Generative Agents Simulation
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CHARACTERS_DIR = os.path.join(ASSETS_DIR, "characters")
SPEECH_BUBBLE_DIR = os.path.join(ASSETS_DIR, "speech_bubble")
VILLE_DIR = os.path.join(ASSETS_DIR, "the_ville")
MAZE_DIR = os.path.join(VILLE_DIR, "matrix")
VISUALS_DIR = os.path.join(VILLE_DIR, "visuals")

# Simulation settings
TIME_STEP_SECONDS = 10  # Each step represents 10 seconds in game time
SIMULATION_SPEED = 2.0  # Steps per second in real time
DAY_DURATION_STEPS = 8640  # 24 hours * 60 minutes * 60 seconds / 10 seconds per step
NUM_AGENTS = 3  # Number of agents to simulate

# Display settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TILE_SIZE = 32  # Size of each tile in pixels
VISION_RADIUS = 8  # How far agents can see

# Agent settings
INTERACTION_DISTANCE = 2  # Tiles
MAX_THOUGHTS_PER_STEP = 1
MOVEMENT_SPEED = 1  # Tiles per step
SMOOTH_MOVE_SPEED = 0.225  # Interpolation speed for smooth movement (0.0 to 1.0)

# Logging
LOG_FILE = "simulation_log.txt"
LOG_ACTIONS = True
LOG_THOUGHTS = True

# Random thought/action lists
RANDOM_THOUGHTS = [
    "I wonder what's happening at the cafe",
    "I should check my schedule",
    "I'm feeling a bit tired",
    "The weather seems nice today",
    "I need to get some work done",
    "Maybe I should visit the park",
    "I remember something I forgot to do",
    "I'm looking forward to the party",
    "I should talk to someone",
    "I wonder where my friends are"
]

RANDOM_ACTIONS = [
    "walking around",
    "standing still",
    "looking around",
    "checking phone",
    "thinking deeply",
    "stretching",
    "observing surroundings"
]

INTERACTION_MESSAGES = [
    "Hi there!",
    "How are you?",
    "Nice to see you",
    "What's up?",
    "Good morning!",
    "Hello!",
    "Hey!",
    "What brings you here?"
]
