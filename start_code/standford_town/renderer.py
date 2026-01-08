"""
Pygame Renderer for Generative Agents Simulation
"""
import os
import random
import pygame
import config


class Renderer:
    def __init__(self, world, agents):
        pygame.init()
        self.world = world
        self.agents = agents

        # Set up display
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        pygame.display.set_caption("Generative Agents - Stanford Town")

        # Camera offset (scrolling)
        self.camera_x = 0
        self.camera_y = 0

        # Animation frame counter
        self.frame_count = 0
        
        # Track animation frame for each agent (0, 1, or 2)
        self.agent_animation_frames = {}
        self.agent_last_direction = {}

        # Colors
        self.colors = {
            'background': (200, 200, 200),
            'tile_empty': (220, 220, 220),
            'tile_blocked': (100, 100, 100),
            'text': (0, 0, 0),
            'agent_name': (255, 255, 255),
            'chat_bubble_bg': (255, 255, 255),
            'chat_bubble_border': (0, 0, 0)
        }

        # Load assets
        self.load_assets()

        # Fonts
        self.font_name = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)

        print("Renderer initialized")

    def load_assets(self):
        """Load character sprites and other assets"""
        self.agent_sprites = {}
        self.agent_images = {}
        self.agent_animations = {}

        # Load character images
        character_files = os.listdir(config.CHARACTERS_DIR)
        for filename in character_files:
            if filename.endswith('.png'):
                name = filename.replace('.png', '')
                path = os.path.join(config.CHARACTERS_DIR, filename)
                try:
                    image = pygame.image.load(path)
                    # Extract animation frames from spritesheet
                    # Format: 96x128 spritesheet with 4 directions (down, left, right, up)
                    # Each direction has 3 frames at x positions (0, 32, 64)
                    # Rows: y=0 (down), y=32 (left), y=64 (right), y=96 (up)
                    self.agent_animations[name] = {
                        'down': [
                            self.extract_frame(image, 0, 0),
                            self.extract_frame(image, 32, 0),
                            self.extract_frame(image, 64, 0)
                        ],
                        'left': [
                            self.extract_frame(image, 0, 32),
                            self.extract_frame(image, 32, 32),
                            self.extract_frame(image, 64, 32)
                        ],
                        'right': [
                            self.extract_frame(image, 0, 64),
                            self.extract_frame(image, 32, 64),
                            self.extract_frame(image, 64, 64)
                        ],
                        'up': [
                            self.extract_frame(image, 0, 96),
                            self.extract_frame(image, 32, 96),
                            self.extract_frame(image, 64, 96)
                        ]
                    }
                    # Default image is down facing, first frame
                    self.agent_images[name] = self.agent_animations[name]['down'][0]
                except Exception as e:
                    print(f"Could not load {filename}: {e}")

        # Load speech bubbles
        self.load_speech_bubbles()

    def extract_frame(self, spritesheet, x, y):
        """Extract a single frame from the spritesheet"""
        # Each frame is 32x32
        frame = pygame.Surface((32, 32), pygame.SRCALPHA)
        frame.blit(spritesheet, (0, 0), (x, y, 32, 32))
        return frame

    def load_speech_bubbles(self):
        """Load speech bubble assets"""
        self.speech_bubbles = {}
        speech_files = os.listdir(config.SPEECH_BUBBLE_DIR)
        for filename in speech_files:
            if filename.endswith('.png'):
                name = filename.replace('.png', '')
                path = os.path.join(config.SPEECH_BUBBLE_DIR, filename)
                try:
                    image = pygame.image.load(path)
                    self.speech_bubbles[name] = image
                except Exception as e:
                    print(f"Could not load {filename}: {e}")

        # Try to load map background
        map_file = os.path.join(config.VISUALS_DIR, "the_ville2.png")
        if os.path.exists(map_file):
            try:
                self.map_background = pygame.image.load(map_file)
                # Scale to fit world dimensions
                self.map_background = pygame.transform.scale(
                    self.map_background,
                    (self.world.width * config.TILE_SIZE,
                     self.world.height * config.TILE_SIZE)
                )
            except Exception as e:
                print(f"Could not load map: {e}")
                self.map_background = None
        else:
            self.map_background = None

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates"""
        screen_x = world_x * config.TILE_SIZE - self.camera_x
        screen_y = world_y * config.TILE_SIZE - self.camera_y
        return screen_x, screen_y

    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_x + self.camera_x) // config.TILE_SIZE
        world_y = (screen_y + self.camera_y) // config.TILE_SIZE
        return world_x, world_y

    def center_camera_on(self, world_x, world_y):
        """Center the camera on a world position"""
        self.camera_x = world_x * config.TILE_SIZE - config.WINDOW_WIDTH // 2
        self.camera_y = world_y * config.TILE_SIZE - config.WINDOW_HEIGHT // 2

    def handle_input(self):
        """Handle keyboard input for camera control"""
        keys = pygame.key.get_pressed()
        camera_speed = 10

        if keys[pygame.K_LEFT]:
            self.camera_x -= camera_speed
        if keys[pygame.K_RIGHT]:
            self.camera_x += camera_speed
        if keys[pygame.K_UP]:
            self.camera_y -= camera_speed
        if keys[pygame.K_DOWN]:
            self.camera_y += camera_speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

        return True

    def draw_background(self):
        """Draw the world background"""
        self.screen.fill(self.colors['background'])

        # Draw map background if available
        if self.map_background:
            screen_x, screen_y = self.world_to_screen(0, 0)
            self.screen.blit(self.map_background, (screen_x, screen_y))
        else:
            # Draw tiles
            for x in range(self.world.width):
                for y in range(self.world.height):
                    screen_x, screen_y = self.world_to_screen(x, y)

                    # Only draw if on screen
                    if -config.TILE_SIZE <= screen_x <= config.WINDOW_WIDTH:
                        if -config.TILE_SIZE <= screen_y <= config.WINDOW_HEIGHT:
                            tile_data = self.world.tiles[y][x]
                            color = self.colors['tile_blocked'] if tile_data['blocked'] else self.colors['tile_empty']
                            pygame.draw.rect(self.screen, color,
                                           (screen_x, screen_y, config.TILE_SIZE, config.TILE_SIZE))
                            pygame.draw.rect(self.screen, (150, 150, 150),
                                           (screen_x, screen_y, config.TILE_SIZE, config.TILE_SIZE), 1)

    def draw_agents(self):
        """Draw all agents with smooth movement"""
        for agent in self.agents:
            # Use interpolated display position for smooth movement
            screen_x, screen_y = self.world_to_screen(agent.display_x, agent.display_y)

            # Only draw if on screen
            if -config.TILE_SIZE <= screen_x <= config.WINDOW_WIDTH:
                if -config.TILE_SIZE <= screen_y <= config.WINDOW_HEIGHT:

                    # Initialize animation state for this agent if needed
                    if agent.name not in self.agent_animation_frames:
                        self.agent_animation_frames[agent.name] = 0
                        self.agent_last_direction[agent.name] = agent.direction
                    
                    # Check if direction changed, reset animation frame to 0
                    if agent.name in self.agent_last_direction:
                        if self.agent_last_direction[agent.name] != agent.direction:
                            self.agent_animation_frames[agent.name] = 0
                            self.agent_last_direction[agent.name] = agent.direction
                    
                    # Get current direction and cycle through 3 animation frames
                    direction = agent.direction
                    
                    # Update animation frame every 10 frames
                    if agent.is_moving and self.frame_count % 10 == 0:
                        self.agent_animation_frames[agent.name] = (self.agent_animation_frames[agent.name] + 1) % 3

                    # Draw agent image if available
                    if agent.name in self.agent_animations:
                        frames = self.agent_animations[agent.name][direction]
                        current_frame = frames[self.agent_animation_frames[agent.name]]
                        self.screen.blit(current_frame, (screen_x, screen_y))
                    elif agent.name in self.agent_images:
                        self.screen.blit(self.agent_images[agent.name], (screen_x, screen_y))
                    else:
                        # Fallback: draw a colored circle (consistent color based on name)
                        hash_val = sum(ord(c) for c in agent.name)
                        color = ((hash_val * 37) % 200 + 55, (hash_val * 73) % 200 + 55, (hash_val * 151) % 200 + 55)
                        pygame.draw.circle(self.screen, color,
                                        (screen_x + config.TILE_SIZE // 2,
                                         screen_y + config.TILE_SIZE // 2),
                                           config.TILE_SIZE // 2 - 2)

                    # Draw pronunciatio (emoji) above agent
                    if agent.pronunciatio:
                        emoji_text = self.font_name.render(agent.pronunciatio, True, (0, 0, 0))
                        self.screen.blit(emoji_text, (screen_x, screen_y - 20))

                    # Draw agent name
                    name_text = self.font_small.render(agent.name, True, self.colors['agent_name'])
                    # Create a background for the name
                    name_rect = name_text.get_rect()
                    name_rect.x = screen_x - name_rect.width // 2 + config.TILE_SIZE // 2
                    name_rect.y = screen_y - 15

                    # Draw chat bubble if agent is chatting
                    if agent.chat_message:
                        bubble_rect = pygame.Rect(screen_x - 30, screen_y - 50, 80, 30)
                        pygame.draw.rect(self.screen, self.colors['chat_bubble_bg'], bubble_rect)
                        pygame.draw.rect(self.screen, self.colors['chat_bubble_border'], bubble_rect, 2)

                        chat_text = self.font_small.render(agent.chat_message, True, (0, 0, 0))
                        chat_rect = chat_text.get_rect(center=bubble_rect.center)
                        self.screen.blit(chat_text, chat_rect)

    def draw_ui(self):
        """Draw UI elements"""
        # Draw simulation info
        time_text = self.font_name.render(f"Time: 00:00 (Step 0)", True, self.colors['text'])
        self.screen.blit(time_text, (10, 10))

        agent_text = self.font_name.render(f"Agents: {len(self.agents)}", True, self.colors['text'])
        self.screen.blit(agent_text, (10, 35))

        controls_text = self.font_small.render("Arrow keys to scroll, ESC to quit", True, (100, 100, 100))
        self.screen.blit(controls_text, (10, config.WINDOW_HEIGHT - 30))

    def render(self, time_str="00:00", step=0):
        """Render the entire scene"""
        # Handle input
        if not self.handle_input():
            return False

        # Increment frame counter for animation
        self.frame_count += 1

        # Draw everything
        self.draw_background()
        self.draw_agents()
        self.draw_ui()

        # Update time display
        pygame.draw.rect(self.screen, self.colors['background'], (10, 10, 200, 60))
        time_text = self.font_name.render(f"Time: {time_str} (Step {step})", True, self.colors['text'])
        self.screen.blit(time_text, (10, 10))

        # Update display
        pygame.display.flip()

        return True

    def cleanup(self):
        """Clean up pygame resources"""
        pygame.quit()
