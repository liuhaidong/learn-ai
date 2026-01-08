"""
Main entry point for Generative Agents Simulation
"""
import sys
import argparse
from simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description='Generative Agents Simulation')
    parser.add_argument('--no-gui', action='store_true',
                       help='Run without GUI (headless mode)')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of steps to run (default: 1 day)')
    parser.add_argument('--speed', type=float, default=2.0,
                       help='Simulation speed in steps/second')
    args = parser.parse_args()

    print("=" * 80)
    print("Generative Agents: Interactive Simulacra of Human Behavior")
    print("=" * 80)
    print("\nThis is a demo implementation inspired by the Stanford paper.")
    print("Agents use random thoughts and actions (no LLM API calls).")
    print("\nFeatures:")
    print("  - 3 autonomous agents with memory systems")
    print("  - 2D world based on Smallville map data")
    print("  - Real-time visualization with pygame")
    print("  - Agent interactions and conversations")
    print("  - Memory, reflection, and planning systems")
    print("\nControls:")
    print("  - Arrow keys: Scroll camera")
    print("  - ESC: Quit simulation")
    print("=" * 80)

    # Create and run simulation
    sim = Simulation()
    sim.run()


if __name__ == "__main__":
    main()
