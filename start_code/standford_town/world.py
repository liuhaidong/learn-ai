"""
World/Environment System for Generative Agents
"""
import os
import csv
import heapq
import random
import config


class World:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.tiles = []  # 2D array of tile data
        self.collision_map = {}  # (x, y) -> bool (True if blocked)
        self.sector_map = {}  # (x, y) -> sector name
        self.arena_map = {}  # (x, y) -> arena name
        self.game_object_map = {}  # (x, y) -> game object name

        self.load_maze_data()

    def load_maze_data(self):
        """Load maze data - simplified version"""
        print("Loading maze data...")

        # Create a simple world with basic collision pattern
        # Smaller world size (20x20) to better match character scale
        self.width = 20
        self.height = 20

        # Create collision map (walls around edges, some internal obstacles)
        for y in range(self.height):
            for x in range(self.width):
                # Walls around edges
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.collision_map[(x, y)] = True
                # Some internal obstacles (buildings)
                elif (3 < x < 8 and 7 < y < 13) or (12 < x < 17 and 10 < y < 16):
                    self.collision_map[(x, y)] = True
                else:
                    self.collision_map[(x, y)] = False

        # Initialize tile array
        self.tiles = [[{} for _ in range(self.width)] for _ in range(self.height)]

        # Load sector data
        sector_file = os.path.join(config.MAZE_DIR, "sector_maze.csv")
        if os.path.exists(sector_file):
            with open(sector_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            x = int(row[0])
                            y = int(row[1])
                            sector = row[2].strip()
                            if sector:
                                self.sector_map[(x, y)] = sector
                        except ValueError:
                            continue

        # Load arena data
        arena_file = os.path.join(config.MAZE_DIR, "arena_maze.csv")
        if os.path.exists(arena_file):
            with open(arena_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            x = int(row[0])
                            y = int(row[1])
                            arena = row[2].strip()
                            if arena:
                                self.arena_map[(x, y)] = arena
                        except ValueError:
                            continue

        # Load game object data
        game_object_file = os.path.join(config.MAZE_DIR, "game_object_maze.csv")
        if os.path.exists(game_object_file):
            with open(game_object_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            x = int(row[0])
                            y = int(row[1])
                            game_object = row[2].strip()
                            if game_object:
                                self.game_object_map[(x, y)] = game_object
                        except ValueError:
                            continue

        # Initialize tile array
        self.tiles = [[{} for _ in range(self.width)] for _ in range(self.height)]

        # Populate tile data
        for x in range(self.width):
            for y in range(self.height):
                self.tiles[y][x] = {
                    'blocked': self.collision_map.get((x, y), False),
                    'sector': self.sector_map.get((x, y), ""),
                    'arena': self.arena_map.get((x, y), ""),
                    'game_object': self.game_object_map.get((x, y), ""),
                    'events': set()
                }

        print(f"World loaded: {self.width}x{self.height} tiles")

    def is_valid_tile(self, x, y):
        """Check if a tile is valid (within bounds and not blocked)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return not self.collision_map.get((x, y), False)
        return False

    def get_tile_name(self, x, y):
        """Get a descriptive name for a tile"""
        sector = self.sector_map.get((x, y), "")
        arena = self.arena_map.get((x, y), "")
        game_object = self.game_object_map.get((x, y), "")

        parts = []
        if sector:
            parts.append(sector)
        if arena and arena != sector:
            parts.append(arena)
        if game_object:
            parts.append(f"at {game_object}")

        return " > ".join(parts) if parts else f"Tile ({x}, {y})"

    def add_event_to_tile(self, x, y, event):
        """Add an event to a tile"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.tiles[y][x]['events'].add(event)

    def remove_event_from_tile(self, x, y, event):
        """Remove an event from a tile"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.tiles[y][x]['events'].discard(event)

    def get_nearby_tiles(self, x, y, radius):
        """Get all tiles within a radius"""
        nearby = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if self.is_valid_tile(nx, ny):
                    nearby.append((nx, ny))
        return nearby

    def find_path(self, start, end):
        """
        Find a path from start to end using A* algorithm
        start, end: (x, y) tuples
        Returns: list of (x, y) tuples representing the path
        """
        if not self.is_valid_tile(end[0], end[1]):
            return []

        # Priority queue for A*
        open_set = []
        heapq.heappush(open_set, (0, start))

        # Track best path to each node
        came_from = {}
        g_score = {start: 0}

        # Heuristic function (Manhattan distance)
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            # Explore neighbors (4 directions)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.is_valid_tile(neighbor[0], neighbor[1]):
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

        return []  # No path found

    def get_random_valid_tile(self):
        """Get a random valid tile in the world"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.is_valid_tile(x, y):
                return (x, y)
        return (0, 0)  # Fallback

    def get_spawn_positions(self, num_positions):
        """Get spawn positions from the spawn file or random valid tiles"""
        spawn_positions = []
        spawn_file = os.path.join(config.MAZE_DIR, "spawning_location_maze.csv")

        if os.path.exists(spawn_file):
            with open(spawn_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            x = int(row[0])
                            y = int(row[1])
                            if self.is_valid_tile(x, y):
                                spawn_positions.append((x, y))
                        except ValueError:
                            continue

        # If not enough spawn positions, use random valid tiles
        while len(spawn_positions) < num_positions:
            spawn_positions.append(self.get_random_valid_tile())

        return spawn_positions[:num_positions]

    def print_world_info(self):
        """Print world information"""
        print(f"\n=== World Information ===")
        print(f"Dimensions: {self.width}x{self.height}")
        print(f"Total tiles: {self.width * self.height}")
        print(f"Blocked tiles: {len([k for k, v in self.collision_map.items() if v])}")

        # Count sectors
        sectors = set(self.sector_map.values())
        print(f"Sectors: {len(sectors)}")

        # Count arenas
        arenas = set(self.arena_map.values())
        print(f"Arenas: {len(arenas)}")
