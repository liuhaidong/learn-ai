"""
Auto-run demo for Generative Agents (headless)
Simple version without complex features
"""
import os
from datetime import datetime, timedelta
import random


# Configuration
TIME_STEP_SECONDS = 10
SIMULATION_SPEED = 2.0
NUM_AGENTS = 3
DAY_DURATION_STEPS = 864  # 24 hours

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CHARACTERS_DIR = os.path.join(ASSETS_DIR, "characters")


class SimpleAgent:
    """Simplified agent for demo"""
    def __init__(self, name, x, y, sprite_path):
        self.name = name
        self.x = x
        self.y = y
        self.sprite_path = sprite_path
        self.thought = ""
        self.action = "idle"
        self.chat_message = ""
        self.chatting_with = None
        self.memories = []

    def generate_thought(self):
        """Generate a random thought"""
        thoughts = [
            "I wonder what's happening at cafe",
            "I should check my schedule",
            "I'm feeling a bit tired",
            "The weather seems nice today",
            "I need to get some work done",
            "Maybe I should visit the park",
            "I remember something I forgot to do",
            "I should talk to someone",
            "I'm looking forward to the party",
            "Hello!",
            "Hey!",
            "What's up?",
            "Good morning!",
            "Nice to see you",
            "Want to do something fun?",
            "It's nice weather",
            "I've got a joke for you",
            "You look busy",
            "Should we hang out later?"
            "Remember our plans!"
        ]
        return random.choice(thoughts)

    def choose_action(self):
        """Choose a random action"""
        actions = [
            "walking around",
            "standing still",
            "looking around",
            "checking phone",
            "thinking deeply",
            "stretching",
            "observing surroundings",
            "talking to self",
            "reading a book",
            "writing notes",
            "having lunch",
            "waiting for someone"
        ]
        self.action = random.choice(actions)

    def move_randomly(self):
        """Move randomly to a nearby tile"""
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_x = max(0, min(self.x + dx, 63))
        new_y = max(0, min(self.y + dy, 63))
        self.x = new_x
        self.y = new_y
        self.chatting_with = None
        self.chat_message = ""

    def interact(self, other_agents):
        """Try to interact with another agent"""
        if random.random() < 0.3:  # 30% chance
            other_agent = random.choice(other_agents)
            if not other_agent.chatting_with:
                messages = [
                    "Hi there!",
                    "How are you?",
                    "Nice to see you",
                    "What's up?",
                    "Good morning!",
                    "Hello!",
                    "Hey!",
                    "What brings you here?",
                    "Want to do something fun?",
                    "It's nice weather",
                    "I've got a joke for you",
                    "You look busy",
                    "Should we hang out later?",
                    "Remember our plans!"
                ]
                self.chat_message = random.choice(messages)
                other_agent.chatting_with = self.name
                self.chatting_with = other_agent.name


class SimpleWorld:
    """Simplified world for demo"""
    def __init__(self):
        # Create a simple 64x64 world with some obstacles
        self.width = 64
        self.height = 64

        # Create tiles
        self.tiles = {}
        for y in range(self.height):
            for x in range(self.width):
                # Walls around edges
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.tiles[(x, y)] = {"blocked": True}
                # Some obstacles in the middle
                elif (10 < x < 20 and 20 < y < 30) or (40 < x < 50 and 40 < y < 50):
                    self.tiles[(x, y)] = {"blocked": True}
                # Most tiles are walkable
                else:
                    self.tiles[(x, y)] = {"blocked": False}

    def is_valid_tile(self, x, y):
        """Check if a tile is valid (within bounds and not blocked)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return not self.tiles[(x, y)]["blocked"]
        return False

    def get_random_valid_tile(self):
        """Get a random valid tile"""
        for _ in range(100):  # Max attempts
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.is_valid_tile(x, y):
                return x, y
        return None

    def get_spawn_positions(self, num_positions):
        """Get random spawn positions"""
        positions = []
        max_attempts = 100

        for i in range(num_positions):
            pos = self.get_random_valid_tile()
            if pos and pos not in positions:  # Avoid duplicates
                positions.append(pos)
            if len(positions) >= num_positions:
                break

        return positions[:num_positions]

    def get_nearby_tiles(self, x, y, radius=2):
        """Get nearby tiles within radius"""
        nearby = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if self.is_valid_tile(nx, ny):
                    nearby.append((nx, ny))
        return nearby

    def get_tile_name(self, x, y):
        """Get a descriptive name for a tile"""
        return f"Tile ({x}, {y})"


def run_demo(num_steps=100, num_agents=3):
    """Run auto demo simulation"""
    print("=" * 80)
    print("GENERATIVE AGENTS AUTO-RUN (HEADLESS)")
    print("=" * 80)
    print(f"\nRunning for {num_steps} steps ({num_steps * TIME_STEP_SECONDS}s game time)")
    print("This is a demo with random thoughts/actions (no LLM API)\n")

    # Initialize world
    print("\nInitializing...")
    world = SimpleWorld()
    print(f"World created: {world.width}x{world.height}")

    # Get spawn positions
    spawn_positions = world.get_spawn_positions(num_agents)
    character_files = os.listdir(CHARACTERS_DIR)
    character_files = [f for f in character_files if f.endswith('.png')]

    # Create agents
    agents = []
    for i in range(min(num_agents, len(character_files), len(spawn_positions))):
        name = character_files[i].replace('.png', '').replace('_', ' ')
        x, y = spawn_positions[i]
        sprite_path = os.path.join(CHARACTERS_DIR, character_files[i])
        agent = SimpleAgent(name, x, y, sprite_path)
        agents.append(agent)
        print(f"Created {name} at ({x}, {y})")

    # Set up time
    start_time = datetime(2024, 1, 1, 7, 0, 0)  # 7 AM
    current_time = start_time
    print(f"\nSimulation time: {start_time.strftime('%Y-%m-%d %H:%M')}")

    # Run simulation
    print("\n" + "=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)

    step_count = 0
    conversation_count = 0

    for step in range(num_steps):
        step_count += 1
        current_time += timedelta(seconds=TIME_STEP_SECONDS)
        time_str = current_time.strftime('%H:%M')

        # Clear chat messages periodically
        if step_count % 10 == 0:
            for agent in agents:
                agent.chat_message = ""

        # Update each agent
        for agent in agents:
            # Generate thought
            agent.thought = agent.generate_thought()

            # Choose action
            agent.action = agent.choose_action()

            # Move randomly
            agent.move_randomly()

            # Find nearby agents
            nearby_agents = []
            for other_agent in agents:
                dx = other_agent.x - agent.x
                dy = other_agent.y - agent.y
                if abs(dx) <= 2 and abs(dy) <= 2:
                    nearby_agents.append(other_agent)

            # Interaction chance
            if nearby_agents and random.random() < 0.2:  # 20% chance
                agent.interact(nearby_agents)

            # Log interesting events
            if agent.chat_message:
                conversation_count += 1
                print(f"[{time_str}] {agent.name}: \"{agent.chat_message}\"")

            elif agent.thought and step_count % 5 == 0:
                print(f"[{time_str}] {agent.name} thinks: {agent.thought}")

            elif step_count % 3 == 0 and agent.action != "idle":
                print(f"[{time_str}] {agent.name}: {agent.action}")

            elif step_count % 2 == 0:
                pass  # Reduce log spam

        # Progress indicator
        if step_count % 20 == 0:
            progress = (step_count / num_steps) * 100
            bar_length = 50
            filled = int(progress / 2)
            spaces = ' ' * (bar_length - filled)
            print(f"[{'█' * filled}{spaces}] {progress:.0f}% - Step {step_count}/{num_steps}")

    # Final summary
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal steps: {step_count}")
    print(f"Final time: {current_time.strftime('%H:%M')}")
    print(f"Conversations: {conversation_count}")

    for agent in agents:
        print(f"\n{agent.name}:")
        print(f"  Final position: ({agent.x}, {agent.y})")
        print(f" Total memories: {len(agent.memories)}")
        recent_memories = agent.memories[-5:] if len(agent.memories) >= 5 else agent.memories
        for mem in recent_memories:
            print(f"  - {mem}")

    print("\n✓ Simulation completed successfully!")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Generative Agents demo (headless)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='%(prog)s'
    )
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of steps to simulate (default: 100)')
    parser.add_argument('--agents', type=int, default=3,
                       help='Number of agents (default: 3)')

    args = parser.parse_args()

    # Override number of agents if specified
    num_agents = args.agents if args.agents else NUM_AGENTS

    run_demo(args.steps, num_agents)


if __name__ == "__main__":
    main()
