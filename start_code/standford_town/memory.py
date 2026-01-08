"""
Memory System for Generative Agents
Simplified version using random selection instead of embeddings
"""
import time
import random
from datetime import datetime, timedelta


class Memory:
    """Represents a single memory entry"""
    def __init__(self, content, memory_type="event", importance=1):
        self.content = content
        self.memory_type = memory_type  # event, thought, chat
        self.importance = importance  # 1-10 scale
        self.timestamp = time.time()
        self.created_at = datetime.now()


class MemoryStream:
    """Associative memory stream for agents"""
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.memories = []
        self.reflection_threshold = 50  # Importance sum threshold for reflection

    def add_memory(self, content, memory_type="event", importance=None):
        """Add a new memory to the stream"""
        if importance is None:
            importance = random.randint(1, 5)

        memory = Memory(content, memory_type, importance)
        self.memories.append(memory)
        return memory

    def retrieve_memories(self, query, k=5):
        """
        Retrieve k memories relevant to the query
        Simplified: uses random selection weighted by recency and importance
        """
        if not self.memories:
            return []

        # Calculate scores based on recency, importance
        scored_memories = []
        current_time = time.time()

        for memory in self.memories:
            # Recency: more recent = higher score
            age_seconds = current_time - memory.timestamp
            recency_score = max(0, 1 - (age_seconds / 86400))  # Decay over a day

            # Importance: 1-10 scale
            importance_score = memory.importance / 10

            # Total score (0-1 range)
            total_score = (recency_score * 0.5) + (importance_score * 0.5)
            scored_memories.append((memory, total_score))

        # Sort by score and return top k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in scored_memories[:k]]

    def reflect(self):
        """Generate reflection if threshold met"""
        recent_importance_sum = sum(
            m.importance for m in self.memories[-10:]
        )

        if recent_importance_sum >= self.reflection_threshold:
            # Generate a reflection (simplified: random thought about recent experiences)
            recent_memories = self.memories[-5:]
            if recent_memories:
                reflection = f"Reflecting on my recent experiences..."
                self.add_memory(reflection, memory_type="thought", importance=7)
                return reflection
        return None

    def get_total_importance(self):
        """Get sum of recent memory importances"""
        return sum(m.importance for m in self.memories[-10:])

    def print_summary(self):
        """Print summary of memories"""
        print(f"\n=== {self.agent_name}'s Memory Summary ===")
        print(f"Total memories: {len(self.memories)}")
        print(f"Recent thoughts:")
        for mem in self.memories[-5:]:
            print(f"  - {mem.content} (importance: {mem.importance})")
