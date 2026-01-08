"""
Test script to verify core functionality without GUI
"""
import sys
from world import World
from memory import MemoryStream
from agent import Agent


def test_world():
    """Test world loading"""
    print("Testing World loading...")
    world = World()
    print(f"  World size: {world.width}x{world.height}")
    print(f"  Valid tiles: {sum(1 for x in range(world.width) for y in range(world.height) if world.is_valid_tile(x, y))}")
    print("  ✓ World loaded successfully\n")


def test_memory():
    """Test memory system"""
    print("Testing Memory system...")
    memory = MemoryStream("TestAgent")

    # Add memories
    memory.add_memory("I went to the cafe", "event", importance=5)
    memory.add_memory("I had coffee", "event", importance=3)
    memory.add_memory("I felt tired", "thought", importance=2)

    # Retrieve memories
    relevant = memory.retrieve_memories("cafe", k=2)
    print(f"  Added 3 memories, retrieved {len(relevant)} relevant")

    # Test reflection
    for i in range(20):
        memory.add_memory(f"Memory {i}", "event", importance=3)

    reflection = memory.reflect()
    print(f"  Reflection after adding more memories: {reflection is not None}")

    print("  ✓ Memory system working\n")


def test_agent():
    """Test agent creation and basic operations"""
    print("Testing Agent creation...")
    world = World()
    spawn_pos = world.get_random_valid_tile()

    agent = Agent("Test Agent", spawn_pos[0], spawn_pos[1], "test.png")
    agent.initialize_schedule()

    print(f"  Agent name: {agent.name}")
    print(f"  Agent position: {agent.current_tile}")
    print(f"  Schedule length: {len(agent.daily_schedule)}")
    print(f"  First activity: {agent.daily_schedule[0]}")
    print(f"  Last activity: {agent.daily_schedule[-1]}")

    # Test thought generation
    thought = agent.generate_random_thought()
    print(f"  Generated thought: {thought}")

    # Test action choice
    action = agent.choose_random_action()
    print(f"  Chose action: {action}")

    print("  ✓ Agent working\n")


def test_pathfinding():
    """Test pathfinding"""
    print("Testing Pathfinding...")
    world = World()

    # Find two valid tiles
    start = world.get_random_valid_tile()
    end = world.get_random_valid_tile()

    print(f"  Finding path from {start} to {end}...")

    path = world.find_path(start, end)

    if path:
        print(f"  ✓ Found path with {len(path)} steps")
        # Verify path is valid
        for tile in path:
            assert world.is_valid_tile(tile[0], tile[1]), f"Invalid tile in path: {tile}"
        print(f"  ✓ Path verified")
    else:
        print(f"  ! No path found (tiles may be disconnected)")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Core Functionality Tests")
    print("=" * 60)
    print()

    try:
        test_world()
        test_memory()
        test_agent()
        test_pathfinding()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
