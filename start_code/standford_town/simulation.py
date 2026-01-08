"""
Main Simulation System for Generative Agents
"""
import time
from datetime import datetime, timedelta
import pygame
import config
from world import World
from agent import Agent
from renderer import Renderer


class Simulation:
    def __init__(self):
        print("Initializing Generative Agents Simulation...")

        # Initialize world
        self.world = World()
        self.world.print_world_info()

        # Initialize agents
        self.agents = []
        self.initialize_agents()

        # Initialize renderer
        self.renderer = Renderer(self.world, self.agents)

        # Simulation state
        self.current_step = 0
        self.start_time = datetime(2024, 1, 1, 7, 0, 0)  # Start at 7 AM
        self.current_time = self.start_time
        self.running = False
        self.finished = False

        # Logging
        self.log_file = open(config.LOG_FILE, 'w', encoding='utf-8')
        self.log("=" * 80)
        self.log(f"Simulation Started: {datetime.now()}")
        self.log("=" * 80)

    def initialize_agents(self):
        """Initialize agents with sprites and positions"""
        print("Initializing agents...")

        # Get agent character files
        import os
        character_files = os.listdir(config.CHARACTERS_DIR)
        character_files = [f for f in character_files if f.endswith('.png')]

        # Get spawn positions
        spawn_positions = self.world.get_spawn_positions(config.NUM_AGENTS)

        # Create agents
        for i in range(min(config.NUM_AGENTS, len(character_files), len(spawn_positions))):
            name = character_files[i].replace('.png', '')
            x, y = spawn_positions[i]
            sprite_path = os.path.join(config.CHARACTERS_DIR, character_files[i])

            agent = Agent(name, x, y, sprite_path)
            agent.initialize_schedule()

            self.agents.append(agent)
            print(f"  Created agent: {name} at ({x}, {y})")

        print(f"Total agents created: {len(self.agents)}")

    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_file.write(log_message + "\n")
        self.log_file.flush()

    def update(self):
        """Update simulation by one step"""
        if self.current_step >= config.DAY_DURATION_STEPS:
            self.finished = True
            return False

        # Update simulation time
        self.current_time += timedelta(seconds=config.TIME_STEP_SECONDS)

        # Update all agents
        agent_updates = {}
        for agent in self.agents:
            update_data = agent.update(self.world, self.agents, self.current_time.strftime("%H:%M"))
            agent_updates[agent.name] = update_data

        # Log updates
        if self.current_step % 10 == 0:  # Log every 10 steps
            self.log(f"\n--- Step {self.current_step} at {self.current_time.strftime('%H:%M')} ---")
            for agent_name, update in agent_updates.items():
                self.log(f"  {agent_name}: {update['action']} ({update['thought']})")
                if update['chat_message']:
                    self.log(f"    Chat: {update['chat_message']}")

        self.current_step += 1
        return True

    def run(self):
        """Run the simulation loop"""
        print("\n=== Starting Simulation ===")
        print(f"Duration: {config.DAY_DURATION_STEPS} steps (1 virtual day)")
        print(f"Speed: {config.SIMULATION_SPEED} steps/second")
        print("Press ESC to stop early")

        self.running = True

        last_update_time = time.time()
        frame_count = 0

        try:
            while self.running and not self.finished:
                # Handle rendering
                time_str = self.current_time.strftime("%H:%M")
                if not self.renderer.render(time_str, self.current_step):
                    self.running = False
                    break

                # Update simulation at configured speed
                current_time = time.time()
                time_since_last_update = current_time - last_update_time

                if time_since_last_update >= 1.0 / config.SIMULATION_SPEED:
                    if not self.update():
                        break
                    last_update_time = current_time

                # Limit frame rate
                pygame.time.delay(int(1000 / 60))  # 60 FPS

                frame_count += 1

                # Print progress every 100 steps
                if frame_count % 100 == 0:
                    progress = (self.current_step / config.DAY_DURATION_STEPS) * 100
                    print(f"Progress: {progress:.1f}% (Step {self.current_step}/{config.DAY_DURATION_STEPS})")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")

        except Exception as e:
            print(f"Error during simulation: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up simulation resources"""
        print("\n=== Simulation Finished ===")

        # Print statistics
        self.log("\n=== Final Statistics ===")
        self.log(f"Total steps: {self.current_step}")
        self.log(f"Final time: {self.current_time.strftime('%H:%M')}")

        for agent in self.agents:
            self.log(f"\n{agent.name}:")
            self.log(f"  Position: {agent.current_tile}")
            self.log(f"  Total memories: {len(agent.memory.memories)}")

            # Print recent memories
            recent_memories = agent.memory.memories[-5:]
            if recent_memories:
                self.log("  Recent memories:")
                for mem in recent_memories:
                    self.log(f"    - {mem.content} ({mem.memory_type}, importance: {mem.importance})")

        # Close log file
        self.log_file.close()
        print(f"\nLog saved to: {config.LOG_FILE}")

        # Clean up renderer
        self.renderer.cleanup()

        print("Simulation ended successfully")


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
