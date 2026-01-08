"""
Agent class for Generative Agents
"""
import random
from datetime import datetime, timedelta
from memory import MemoryStream
import config


class Agent:
    def __init__(self, name, x, y, sprite_path):
        self.name = name
        self.x = x
        self.y = y
        self.sprite_path = sprite_path

        # Memory system
        self.memory = MemoryStream(name)

        # Current state
        self.current_tile = (x, y)
        self.target_tile = None
        self.path = []

        # Smooth movement
        self.display_x = float(x)
        self.display_y = float(y)
        self.is_moving = False
        self.move_speed = config.SMOOTH_MOVE_SPEED  # Speed of interpolation (0.0 to 1.0)
        
        # Direction tracking for animation
        self.direction = 'down'  # Current facing direction: 'down', 'left', 'right', 'up'
        self.last_position = (x, y)
        self.animation_frame = 0  # Current animation frame (0, 1, or 2)

        # Current action/thought
        self.current_action = "idle"
        self.current_thought = ""
        self.chatting_with = None
        self.chat_message = ""

        # Planning
        self.daily_schedule = []

        # Visual
        self.pronunciatio = ""  # Emoji for current action

    def initialize_schedule(self):
        """Generate a random daily schedule"""
        time_slots = []
        activities = [
            ("waking up", "bedroom"),
            ("getting ready", "bathroom"),
            ("eating breakfast", "kitchen"),
            ("going to work", "outside"),
            ("working", "office"),
            ("taking a break", "cafe"),
            ("having lunch", "restaurant"),
            ("continuing work", "office"),
            ("going home", "outside"),
            ("relaxing", "living room"),
            ("eating dinner", "kitchen"),
            ("watching TV", "living room"),
            ("getting ready for bed", "bathroom"),
            ("sleeping", "bedroom")
        ]

        start_hour = 7  # 7 AM
        for i, (activity, location) in enumerate(activities):
            time_slots.append({
                "activity": activity,
                "location": location,
                "time": (start_hour + i * 1) % 24
            })

        self.daily_schedule = time_slots
        return self.daily_schedule

    def perceive(self, world, nearby_agents):
        """Perceive the environment and nearby agents"""
        # Store perception in memory
        if nearby_agents:
            perception = f"Saw agents: {', '.join([a.name for a in nearby_agents])}"
            self.memory.add_memory(perception, importance=3)

    def retrieve_relevant_memories(self, current_situation):
        """Retrieve memories relevant to current situation"""
        return self.memory.retrieve_memories(current_situation, k=3)

    def generate_random_thought(self):
        """Generate a random thought based on context"""
        thought = random.choice(config.RANDOM_THOUGHTS)
        self.current_thought = thought
        self.memory.add_memory(f"Thought: {thought}", memory_type="thought", importance=2)
        return thought

    def choose_random_action(self):
        """Choose a random action"""
        action = random.choice(config.RANDOM_ACTIONS)
        self.current_action = action

        # Set emoji based on action
        emojis = {
            "walking around": "🚶",
            "standing still": "🧍",
            "looking around": "👀",
            "checking phone": "📱",
            "thinking deeply": "🤔",
            "stretching": "🙆",
            "observing surroundings": "🔍"
        }
        self.pronunciatio = emojis.get(action, "🙂")

        return action

    def plan_next_move(self, world, all_agents):
        """Plan the next movement"""
        # If no path, choose a random nearby tile
        if not self.path:
            # Get valid nearby tiles
            possible_moves = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = self.x + dx
                    new_y = self.y + dy
                    if world.is_valid_tile(new_x, new_y):
                        possible_moves.append((new_x, new_y))

            if possible_moves:
                # Prefer moving towards agents (random for now)
                self.target_tile = random.choice(possible_moves)
                self.path = [self.target_tile]

        return self.path

    def move(self, world):
        """Execute movement along the path with smooth interpolation"""
        if self.path:
            next_tile = self.path[0]
            if world.is_valid_tile(next_tile[0], next_tile[1]):
                # Update direction based on movement
                move_dx = next_tile[0] - self.x
                move_dy = next_tile[1] - self.y
                
                if move_dx > 0:
                    self.direction = 'right'
                elif move_dx < 0:
                    self.direction = 'left'
                elif move_dy > 0:
                    self.direction = 'down'
                elif move_dy < 0:
                    self.direction = 'up'
                
                # Smoothly interpolate towards target
                dx = next_tile[0] - self.display_x
                dy = next_tile[1] - self.display_y
                distance = (dx ** 2 + dy ** 2) ** 0.5

                if distance < self.move_speed:
                    # Reached the tile
                    self.x, self.y = next_tile
                    self.display_x = float(next_tile[0])
                    self.display_y = float(next_tile[1])
                    self.path.pop(0)
                    self.current_tile = (self.x, self.y)
                    self.is_moving = False
                    return True
                else:
                    # Continue moving towards target
                    self.display_x += dx * self.move_speed
                    self.display_y += dy * self.move_speed
                    self.is_moving = True
                    return True
        else:
            self.is_moving = False
        return False

    def interact_with_agents(self, nearby_agents):
        """Interact with nearby agents"""
        if nearby_agents and random.random() < 0.3:  # 30% chance to interact
            other_agent = random.choice(nearby_agents)

            # Check if already chatting
            if not other_agent.chatting_with:
                # Start a conversation
                message = random.choice(config.INTERACTION_MESSAGES)
                self.chatting_with = other_agent.name
                other_agent.chatting_with = self.name
                other_agent.chat_message = ""

                self.chat_message = message

                # Log to memory
                self.memory.add_memory(
                    f"Chatted with {other_agent.name}: {message}",
                    memory_type="chat",
                    importance=5
                )
                return True
        return False

    def update(self, world, all_agents, current_time):
        """Main update loop for the agent"""
        # Get nearby agents
        nearby_agents = [
            agent for agent in all_agents
            if agent != self and
            abs(agent.x - self.x) <= config.INTERACTION_DISTANCE and
            abs(agent.y - self.y) <= config.INTERACTION_DISTANCE
        ]

        # Perception
        self.perceive(world, nearby_agents)

        # Retrieve relevant memories
        relevant_memories = self.retrieve_relevant_memories(
            f"Current time: {current_time}, Location: {world.get_tile_name(self.x, self.y)}"
        )

        # Check for reflection
        reflection = self.memory.reflect()
        if reflection:
            if config.LOG_THOUGHTS:
                print(f"[{current_time}] {self.name} REFLECTED: {reflection}")

        # Generate random thought
        thought = self.generate_random_thought()
        if config.LOG_THOUGHTS:
            print(f"[{current_time}] {self.name} thought: {thought}")

        # Choose random action
        action = self.choose_random_action()
        if config.LOG_ACTIONS:
            print(f"[{current_time}] {self.name} action: {action}")

        # Plan movement
        self.plan_next_move(world, all_agents)

        # Execute movement
        moved = self.move(world)
        if moved:
            self.memory.add_memory(
                f"Moved to tile {self.current_tile}",
                memory_type="event",
                importance=1
            )

        # Interact with agents
        self.interact_with_agents(nearby_agents)

        return {
            "thought": thought,
            "action": action,
            "moved": moved,
            "chat_message": self.chat_message
        }

    def get_status(self):
        """Get current agent status"""
        return {
            "name": self.name,
            "position": (self.x, self.y),
            "action": self.current_action,
            "thought": self.current_thought,
            "pronunciatio": self.pronunciatio,
            "chat_message": self.chat_message,
            "chatting_with": self.chatting_with
        }
