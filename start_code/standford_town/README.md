# Generative Agents: Interactive Simulacra of Human Behavior

A demo implementation inspired by the Stanford research paper "Generative Agents: Interactive Simulacra of Human Behavior" (Park et al., 2023).

## Overview

This project simulates autonomous agents in a 2D world (Smallville) with memory, reflection, and planning systems. Agents move, interact, think, and converse using random behaviors (no LLM API calls required for this demo).

## Features

- **Autonomous Agents**: 3 agents with individual personalities, memory systems, and daily schedules
- **2D World**: Based on Smallville map with collision detection, sectors, and game objects
- **Memory System**: Associative memory stream with retrieval by recency and importance
- **Reflection**: Agents reflect on experiences when importance threshold is reached
- **Planning**: Random daily schedules that guide agent behavior
- **Interactions**: Agent-to-agent conversations and proximity-based interactions
- **Visualization**: Real-time pygame rendering with character sprites and speech bubbles
- **Logging**: Detailed logging of agent thoughts, actions, and conversations

## Requirements

```
pygame==2.5.2
```

## Installation

1. Install pygame:
```bash
pip install pygame
```

2. Ensure assets are in place (included in the repository):
```
assets/
├── characters/       # 25 character sprites
├── speech_bubble/    # Speech bubble images
└── the_ville/        # Map and maze data
```

## Usage

Run the simulation:
```bash
python main.py
```

### Controls
- **Arrow keys**: Scroll camera around the map
- **ESC**: Quit simulation

### Configuration

Edit `config.py` to customize:
- `NUM_AGENTS`: Number of agents (default: 3)
- `SIMULATION_SPEED`: Steps per second (default: 2.0)
- `TIME_STEP_SECONDS`: Real-world seconds per game step (default: 10)
- `DAY_DURATION_STEPS`: Steps in a full day (default: 8640)
- `WINDOW_WIDTH`, `WINDOW_HEIGHT`: Display size

## Simulation Details

### Time System
- Each step represents 10 seconds of game time
- Simulation runs for 1 full virtual day (86,400 steps = 24 hours)
- Default speed: 2 steps/second = 20 seconds game time/second real time

### Agent Behavior
1. **Perception**: Agents observe nearby agents and environment
2. **Memory Retrieval**: Recalls relevant past experiences
3. **Reflection**: Thinks about recent experiences if importance threshold reached
4. **Planning**: Generates random daily schedule
5. **Action**: Performs random action (walk, stand, think, etc.)
6. **Movement**: Navigates using A* pathfinding
7. **Interaction**: Converses with nearby agents randomly

### Memory System
- Events: Experiences and observations (importance 1-3)
- Thoughts: Internal monologue (importance 1-2)
- Chats: Conversations with other agents (importance 5)
- Retrieval: Weighted by recency (50%) and importance (50%)

### Output
- **Visual**: Real-time pygame display showing agent movements, actions, and conversations
- **Console**: Progress updates and statistics
- **Log File**: `simulation_log.txt` with detailed record of all agent activities

## Architecture

```
├── main.py              # Entry point
├── simulation.py        # Main simulation loop
├── config.py            # Configuration settings
├── world.py             # World/environment system
├── agent.py             # Agent class with memory, planning
├── memory.py            # Memory stream system
├── renderer.py          # Pygame visualization
└── assets/              # Character sprites, map data
```

## Future Enhancements

- [ ] Integrate LLM API for realistic thoughts and actions
- [ ] Add more complex planning and goal-directed behavior
- [ ] Implement multi-agent coordination and events
- [ ] Add conversation depth and context
- [ ] Support custom scenarios from CSV files
- [ ] Add save/load simulation state
- [ ] Implement agent relationships and social dynamics

## Paper Reference

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23).

https://arxiv.org/abs/2304.03442

## License

This is a demo implementation for educational purposes.
