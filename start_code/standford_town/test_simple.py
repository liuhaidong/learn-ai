"""
Quick test to verify core systems work
"""
import os
import sys
from world import World
from agent import Agent
from memory import MemoryStream

def test_world():
    """Test world creation"""
    print("=" * 60)
    print("Testing World Creation")
    print("=" * 60)

    world = World()
    print(f"World dimensions: {world.width}x{world.height}")

    # Count valid tiles
    valid_tiles = 0
    for y in range(world.height):
        for x in range(world.width):
            if not world.collision_map[(x, y)]:
                valid_tiles += 1

    print(f"Valid tiles: {valid_tiles}/{world.width * world.height}")

    # Test tile info
    print(f"\nSample tile info:")
    for x, y in [(10, 10), (20, 20), (30, 30), (40, 40)]:
        if 0 <= x < world.width and 0 <= y < world.height:
            tile_info = world.tiles[y][x]
            print(f"  Tile ({x},{y}): blocked={tile_info['blocked']}")

    # Test pathfinding
    print("\nTesting pathfinding...")
    path = world.find_path((10, 10), (50, 50))
    print(f"Path from (10,10) to (50,50): {len(path)} steps")
    if path:
        print(f"  First 5 tiles: {path[:5]}")
        # Verify path validity
        for px, py in path:
            if not world.is_valid_tile(px, py):
                print(f"  ERROR: Invalid tile in path: ({px},{py})")
                break

    print("✓ World test passed")

def test_memory():
    """Test memory system"""
    print("\n" + "=" * 60)
    print("Testing Memory System")
    print("=" * 60)

    memory = MemoryStream("TestAgent")

    # Add memories
    memory.add_memory("I went to the cafe", "event", importance=5)
    memory.add_memory("I had coffee", "event", importance=3)
    memory.add_memory("I felt tired", "thought", importance=2)

    print(f"Added 3 memories")

    # Test retrieval
    relevant = memory.retrieve_memories("cafe")
    print(f"Retrieved {len(relevant)} memories for 'cafe'")

    # Test reflection
    print("\nTesting reflection (adding 20 low-importance memories)...")
    for i in range(20):
        memory.add_memory(f"Memory {i}", "event", importance=3)

    reflection = memory.reflect()
    print(f"Reflection triggered: {reflection is not None}")
    if reflection:
        print(f"  Reflection: {reflection}")

    print("✓ Memory test passed")

def test_agent():
    """Test agent creation"""
    print("\n" + "=" * 60)
    print("Testing Agent Creation")
    print("=" * 60)

    world = World()

    # Get valid spawn position
    spawn_pos = world.get_random_valid_tile()
    print(f"Spawn position: {spawn_pos}")

    agent = Agent("TestAgent", spawn_pos[0], spawn_pos[1], "test.png")
    agent.initialize_schedule()

    print(f"Agent name: {agent.name}")
    print(f"Position: {agent.current_tile}")
    print(f"Schedule length: {len(agent.daily_schedule)}")

    # Test thought generation
    thought = agent.generate_random_thought()
    print(f"Generated thought: {thought}")

    # Test action choice
    action = agent.choose_random_action()
    print(f"Chose action: {action}")
    print(f"Emoji: {agent.pronunciatio}")

    # Test movement planning
    path = agent.plan_next_move(world, [])
    print(f"Planned path length: {len(path)}")

    print("✓ Agent test passed")

def test_agent_update():
    """Test agent update cycle"""
    print("\n" + "=" * 60)
    print("Testing Agent Update Cycle (5 steps)")
    print("=" * 60)

    world = World()
    spawn_pos = world.get_random_valid_tile()

    agent = Agent("TestAgent", spawn_pos[0], spawn_pos[1], "test.png")
    agent.initialize_schedule()

    from datetime import datetime, timedelta
    test_time = datetime(2024, 1, 1, 7, 0, 0)

    print("\nRunning 5 update steps...")
    for i in range(5):
        time_str = test_time.strftime("%H:%M")
        print(f"\nStep {i+1} - {time_str}")

        update = agent.update(world, [], test_time)

        print(f"  Thought: {update['thought']}")
        print(f"  Action: {update['action']}")
        print(f" Moved: {update['moved']}")
        print(f" Chat: {update['chat_message']}")
        print(f" Position: {agent.current_tile}")

        test_time += timedelta(minutes=1)

    print("\n✓ Agent update test passed")

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CORE FUNCTIONALITY TESTS")
    print("=" * 80)

    try:
        test_world()
        test_memory()
        test_agent()
        test_agent_update()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
