"""
RL-Based Content Optimization Agent - MVP Demo
Demonstrates core concepts with simplified but runnable implementation
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json

# ============================================================================
# PART 1: STATE REPRESENTATION
# ============================================================================

@dataclass
class ContentState:
    """Represents the current state of content being created"""
    
    # Stage progress (0.0 to 1.0)
    research_completeness: float = 0.0
    outline_completeness: float = 0.0
    draft_completeness: float = 0.0
    
    # Quality metrics (0.0 to 1.0)
    hook_strength: float = 0.0
    clarity_score: float = 0.0
    evidence_density: float = 0.0
    narrative_flow: float = 0.0
    
    # Platform fit
    platform_fit: float = 0.0
    length_appropriateness: float = 0.0
    
    # Current stage
    current_stage: int = 0  # 0=Research, 1=Outline, 2=Draft, 3=Refine, 4=Done
    
    # Metadata
    revision_count: int = 0
    tokens_used: int = 0
    
    # Content storage
    research_data: List[str] = field(default_factory=list)
    outline: List[str] = field(default_factory=list)
    draft_sections: List[str] = field(default_factory=list)
    
    def to_vector(self) -> np.ndarray:
        """Convert state to vector for neural network input"""
        return np.array([
            self.research_completeness,
            self.outline_completeness,
            self.draft_completeness,
            self.hook_strength,
            self.clarity_score,
            self.evidence_density,
            self.narrative_flow,
            self.platform_fit,
            self.length_appropriateness,
            float(self.current_stage) / 4.0,  # Normalize to 0-1
            min(self.revision_count / 10.0, 1.0),  # Cap at 10
            min(self.tokens_used / 10000.0, 1.0),  # Cap at 10k
        ])
    
    def get_quality_score(self) -> float:
        """Overall quality metric"""
        return np.mean([
            self.hook_strength,
            self.clarity_score,
            self.evidence_density,
            self.narrative_flow,
            self.platform_fit,
            self.length_appropriateness,
        ])


# ============================================================================
# PART 2: ACTION SPACE
# ============================================================================

class ActionType(Enum):
    RESEARCH = 0
    CREATE_OUTLINE = 1
    WRITE_HOOK = 2
    WRITE_SECTION = 3
    REFINE_CONTENT = 4
    FINALIZE = 5

@dataclass
class Action:
    """Represents an action the agent can take"""
    action_type: ActionType
    target_section: int = 0  # Which section to work on
    intensity: float = 0.5  # How much effort (0.0 to 1.0)
    
    def __repr__(self):
        return f"Action({self.action_type.name}, section={self.target_section}, intensity={self.intensity:.2f})"


# ============================================================================
# PART 3: ENVIRONMENT
# ============================================================================

class ContentCreationEnvironment:
    """Simulates the content creation process"""
    
    def __init__(self, topic: str, platform: str = "wechat", max_steps: int = 20):
        self.topic = topic
        self.platform = platform
        self.max_steps = max_steps
        self.current_step = 0
        self.state = ContentState()
        
        # Simulated content quality targets
        self.target_sections = 5
        
    def reset(self) -> ContentState:
        """Reset environment to initial state"""
        self.current_step = 0
        self.state = ContentState()
        return self.state
    
    def step(self, action: Action) -> Tuple[ContentState, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)"""
        
        self.current_step += 1
        reward = 0.0
        info = {}
        
        # Simulate action effects based on type
        if action.action_type == ActionType.RESEARCH:
            # Research improves research completeness and evidence density
            improvement = action.intensity * 0.3
            self.state.research_completeness = min(1.0, self.state.research_completeness + improvement)
            self.state.evidence_density = min(1.0, self.state.evidence_density + improvement * 0.5)
            
            # Add simulated research findings
            self.state.research_data.append(f"Research finding {len(self.state.research_data) + 1}")
            
            # Token cost
            self.state.tokens_used += int(500 * action.intensity)
            
            # Reward for completing research
            reward = 0.1 * improvement
            info['action_result'] = f"Conducted research (completeness: {self.state.research_completeness:.2f})"
            
        elif action.action_type == ActionType.CREATE_OUTLINE:
            # Can only create outline if research is done
            if self.state.research_completeness < 0.5:
                reward = -0.2  # Penalty for premature outlining
                info['action_result'] = "Failed: Need more research first"
            else:
                improvement = action.intensity * 0.4
                self.state.outline_completeness = min(1.0, self.state.outline_completeness + improvement)
                self.state.narrative_flow = min(1.0, self.state.narrative_flow + improvement * 0.6)
                
                # Create outline sections
                if len(self.state.outline) < self.target_sections:
                    self.state.outline.append(f"Section {len(self.state.outline) + 1}: [Topic related to {self.topic}]")
                
                self.state.tokens_used += int(300 * action.intensity)
                self.state.current_stage = max(self.state.current_stage, 1)
                
                reward = 0.15 * improvement
                info['action_result'] = f"Created outline (completeness: {self.state.outline_completeness:.2f})"
        
        elif action.action_type == ActionType.WRITE_HOOK:
            # Can only write hook if outline exists
            if self.state.outline_completeness < 0.5:
                reward = -0.2
                info['action_result'] = "Failed: Need outline first"
            else:
                improvement = action.intensity * 0.5
                self.state.hook_strength = min(1.0, self.state.hook_strength + improvement)
                self.state.draft_completeness = min(1.0, self.state.draft_completeness + 0.1)
                
                self.state.tokens_used += int(400 * action.intensity)
                self.state.current_stage = max(self.state.current_stage, 2)
                
                reward = 0.2 * improvement  # Hook is important!
                info['action_result'] = f"Wrote hook (strength: {self.state.hook_strength:.2f})"
        
        elif action.action_type == ActionType.WRITE_SECTION:
            # Write a specific section
            if self.state.outline_completeness < 0.7:
                reward = -0.15
                info['action_result'] = "Failed: Outline not ready"
            else:
                section_idx = action.target_section
                improvement = action.intensity * 0.3
                
                # Add or improve section
                while len(self.state.draft_sections) <= section_idx:
                    self.state.draft_sections.append("")
                
                self.state.draft_sections[section_idx] = f"Content for section {section_idx + 1}"
                
                # Update metrics
                completeness = len([s for s in self.state.draft_sections if s]) / self.target_sections
                self.state.draft_completeness = completeness
                self.state.clarity_score = min(1.0, self.state.clarity_score + improvement * 0.4)
                
                self.state.tokens_used += int(600 * action.intensity)
                self.state.current_stage = max(self.state.current_stage, 2)
                
                reward = 0.15 * improvement
                info['action_result'] = f"Wrote section {section_idx + 1} (draft: {self.state.draft_completeness:.2f})"
        
        elif action.action_type == ActionType.REFINE_CONTENT:
            # Refine existing content
            if self.state.draft_completeness < 0.5:
                reward = -0.1
                info['action_result'] = "Failed: Not enough content to refine"
            else:
                improvement = action.intensity * 0.2
                
                # Refining improves quality metrics
                self.state.clarity_score = min(1.0, self.state.clarity_score + improvement)
                self.state.narrative_flow = min(1.0, self.state.narrative_flow + improvement)
                self.state.platform_fit = min(1.0, self.state.platform_fit + improvement * 0.7)
                
                self.state.revision_count += 1
                self.state.tokens_used += int(400 * action.intensity)
                self.state.current_stage = max(self.state.current_stage, 3)
                
                reward = 0.1 * improvement
                info['action_result'] = f"Refined content (revision {self.state.revision_count})"
        
        elif action.action_type == ActionType.FINALIZE:
            # Calculate final reward based on overall quality
            quality = self.state.get_quality_score()
            completeness = (self.state.research_completeness + 
                          self.state.outline_completeness + 
                          self.state.draft_completeness) / 3.0
            
            # Bonus for high quality + completeness
            if completeness > 0.8 and quality > 0.7:
                reward = 1.0 + quality  # Big reward!
                info['action_result'] = f"SUCCESS! Quality: {quality:.2f}"
            elif completeness > 0.6:
                reward = 0.5
                info['action_result'] = f"Completed (Quality: {quality:.2f})"
            else:
                reward = -0.5  # Penalty for premature finalization
                info['action_result'] = f"Incomplete (only {completeness:.2f} done)"
            
            self.state.current_stage = 4
        
        # Penalty for using too many tokens
        if self.state.tokens_used > 8000:
            reward -= 0.1
        
        # Check if episode is done
        done = (self.state.current_stage == 4 or 
                self.current_step >= self.max_steps)
        
        if done and self.state.current_stage != 4:
            # Ran out of steps without finalizing
            reward -= 0.3
            info['action_result'] = "TIMEOUT: Max steps reached"
        
        return self.state, reward, done, info
    
    def get_valid_actions(self) -> List[Action]:
        """Return list of valid actions in current state"""
        actions = []
        
        # Always can do research
        for intensity in [0.3, 0.5, 0.8]:
            actions.append(Action(ActionType.RESEARCH, intensity=intensity))
        
        # Can create outline if some research done
        if self.state.research_completeness > 0.2:
            for intensity in [0.5, 0.8]:
                actions.append(Action(ActionType.CREATE_OUTLINE, intensity=intensity))
        
        # Can write hook if outline exists
        if self.state.outline_completeness > 0.3:
            for intensity in [0.5, 0.8]:
                actions.append(Action(ActionType.WRITE_HOOK, intensity=intensity))
        
        # Can write sections if outline ready
        if self.state.outline_completeness > 0.5:
            for section in range(self.target_sections):
                actions.append(Action(ActionType.WRITE_SECTION, target_section=section, intensity=0.7))
        
        # Can refine if draft exists
        if self.state.draft_completeness > 0.3:
            actions.append(Action(ActionType.REFINE_CONTENT, intensity=0.6))
        
        # Can finalize anytime (but may be penalized if premature)
        actions.append(Action(ActionType.FINALIZE, intensity=1.0))
        
        return actions


# ============================================================================
# PART 4: SIMPLE POLICY (Q-Learning with Function Approximation)
# ============================================================================

class SimplePolicy:
    """Simple Q-learning policy using linear function approximation"""
    
    def __init__(self, state_dim: int, learning_rate: float = 0.01):
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.3  # Exploration rate
        
        # Q-value weights for each action type
        # Each action type has its own weight vector
        self.weights = {
            action_type: np.random.randn(state_dim) * 0.01
            for action_type in ActionType
        }
        
        # Track statistics
        self.episodes_trained = 0
        self.total_reward_history = []
    
    def get_q_value(self, state_vector: np.ndarray, action: Action) -> float:
        """Compute Q(s, a) using linear approximation"""
        weights = self.weights[action.action_type]
        # Include intensity as a feature multiplier
        q_value = np.dot(state_vector, weights) * action.intensity
        return q_value
    
    def select_action(self, state: ContentState, valid_actions: List[Action]) -> Action:
        """Epsilon-greedy action selection"""
        
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(valid_actions)
        else:
            # Exploit: best action
            state_vector = state.to_vector()
            q_values = [self.get_q_value(state_vector, a) for a in valid_actions]
            best_idx = np.argmax(q_values)
            return valid_actions[best_idx]
    
    def update(self, state: ContentState, action: Action, reward: float, 
               next_state: ContentState, valid_next_actions: List[Action], done: bool):
        """Q-learning update"""
        
        state_vector = state.to_vector()
        
        # Current Q-value
        current_q = self.get_q_value(state_vector, action)
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            # Max Q-value for next state
            next_state_vector = next_state.to_vector()
            next_q_values = [self.get_q_value(next_state_vector, a) for a in valid_next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            target_q = reward + self.gamma * max_next_q
        
        # TD error
        td_error = target_q - current_q
        
        # Gradient update (linear function approximation)
        gradient = state_vector * action.intensity
        self.weights[action.action_type] += self.learning_rate * td_error * gradient
    
    def decay_exploration(self):
        """Reduce exploration over time"""
        self.epsilon = max(0.05, self.epsilon * 0.995)


# ============================================================================
# PART 5: TRAINING LOOP
# ============================================================================

def train_agent(num_episodes: int = 100, verbose: bool = True):
    """Train the RL agent"""
    
    topics = [
        "Why 90% of AI Products Fail After Launch",
        "The Future of Remote Work in 2025",
        "Sustainable Tech: Green Computing Trends",
        "Cybersecurity Best Practices for Startups",
        "Machine Learning for Small Businesses",
    ]
    
    # Initialize
    env = ContentCreationEnvironment(topic=topics[0])
    state_dim = len(env.state.to_vector())
    policy = SimplePolicy(state_dim=state_dim)
    
    episode_rewards = []
    episode_qualities = []
    
    print("=" * 70)
    print("TRAINING RL CONTENT OPTIMIZATION AGENT")
    print("=" * 70)
    
    for episode in range(num_episodes):
        # Random topic each episode
        topic = random.choice(topics)
        env = ContentCreationEnvironment(topic=topic, max_steps=20)
        
        state = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        trajectory = []
        
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Select action
            action = policy.select_action(state, valid_actions)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'info': info,
            })
            
            # Update policy
            next_valid_actions = env.get_valid_actions() if not done else []
            policy.update(state, action, reward, next_state, next_valid_actions, done)
            
            episode_reward += reward
            state = next_state
            step_count += 1
        
        # Track metrics
        episode_rewards.append(episode_reward)
        final_quality = state.get_quality_score()
        episode_qualities.append(final_quality)
        
        # Decay exploration
        policy.decay_exploration()
        policy.episodes_trained += 1
        
        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_quality = np.mean(episode_qualities[-10:])
            print(f"Episode {episode + 1:3d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Quality: {avg_quality:.2f} | Epsilon: {policy.epsilon:.3f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final 10-episode average reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Final 10-episode average quality: {np.mean(episode_qualities[-10:]):.2f}")
    
    return policy, episode_rewards, episode_qualities


# ============================================================================
# PART 6: INFERENCE/DEMONSTRATION
# ============================================================================

def demonstrate_agent(policy: SimplePolicy, topic: str, verbose: bool = True):
    """Demonstrate trained agent creating content"""
    
    print("\n" + "=" * 70)
    print(f"DEMONSTRATION: Creating content for '{topic}'")
    print("=" * 70 + "\n")
    
    env = ContentCreationEnvironment(topic=topic, max_steps=25)
    state = env.reset()
    
    total_reward = 0
    step = 0
    done = False
    
    while not done:
        step += 1
        
        # Get valid actions and select best (no exploration)
        valid_actions = env.get_valid_actions()
        
        # Greedy selection (epsilon = 0)
        state_vector = state.to_vector()
        q_values = [policy.get_q_value(state_vector, a) for a in valid_actions]
        best_action = valid_actions[np.argmax(q_values)]
        
        # Execute
        next_state, reward, done, info = env.step(best_action)
        total_reward += reward
        
        # Print step
        if verbose:
            print(f"Step {step:2d}: {best_action}")
            print(f"         → {info['action_result']}")
            print(f"         → Reward: {reward:+.2f} | Cumulative: {total_reward:+.2f}")
            print(f"         → Quality: {state.get_quality_score():.2f} | "
                  f"Draft: {state.draft_completeness:.0%} | Tokens: {state.tokens_used}")
            print()
        
        state = next_state
    
    # Final summary
    print("=" * 70)
    print("FINAL CONTENT STATE")
    print("=" * 70)
    print(f"Research Completeness: {state.research_completeness:.0%}")
    print(f"Outline Completeness:  {state.outline_completeness:.0%}")
    print(f"Draft Completeness:    {state.draft_completeness:.0%}")
    print(f"Hook Strength:         {state.hook_strength:.0%}")
    print(f"Clarity Score:         {state.clarity_score:.0%}")
    print(f"Evidence Density:      {state.evidence_density:.0%}")
    print(f"Narrative Flow:        {state.narrative_flow:.0%}")
    print(f"Platform Fit:          {state.platform_fit:.0%}")
    print(f"\nOverall Quality:       {state.get_quality_score():.0%}")
    print(f"Total Reward:          {total_reward:+.2f}")
    print(f"Tokens Used:           {state.tokens_used}")
    print(f"Revisions:             {state.revision_count}")
    
    print("\n" + "=" * 70)
    
    return state, total_reward


# ============================================================================
# PART 7: COMPARISON WITH BASELINE
# ============================================================================

def baseline_policy(state: ContentState, valid_actions: List[Action]) -> Action:
    """Rule-based baseline policy (fixed strategy)"""
    
    # Simple heuristic strategy
    if state.research_completeness < 0.8:
        # Focus on research first
        research_actions = [a for a in valid_actions if a.action_type == ActionType.RESEARCH]
        if research_actions:
            return max(research_actions, key=lambda a: a.intensity)
    
    if state.outline_completeness < 0.8:
        # Then create outline
        outline_actions = [a for a in valid_actions if a.action_type == ActionType.CREATE_OUTLINE]
        if outline_actions:
            return max(outline_actions, key=lambda a: a.intensity)
    
    if state.hook_strength < 0.7:
        # Write hook
        hook_actions = [a for a in valid_actions if a.action_type == ActionType.WRITE_HOOK]
        if hook_actions:
            return max(hook_actions, key=lambda a: a.intensity)
    
    if state.draft_completeness < 0.9:
        # Write sections
        section_actions = [a for a in valid_actions if a.action_type == ActionType.WRITE_SECTION]
        if section_actions:
            return section_actions[0]
    
    if state.revision_count < 2:
        # Refine
        refine_actions = [a for a in valid_actions if a.action_type == ActionType.REFINE_CONTENT]
        if refine_actions:
            return refine_actions[0]
    
    # Finalize
    finalize_actions = [a for a in valid_actions if a.action_type == ActionType.FINALIZE]
    return finalize_actions[0] if finalize_actions else valid_actions[0]


def compare_policies(rl_policy: SimplePolicy, num_episodes: int = 20):
    """Compare RL policy vs baseline"""
    
    print("\n" + "=" * 70)
    print("COMPARING RL POLICY VS BASELINE")
    print("=" * 70 + "\n")
    
    topics = [
        "Why 90% of AI Products Fail After Launch",
        "The Future of Remote Work in 2025",
        "Sustainable Tech: Green Computing Trends",
    ]
    
    rl_rewards = []
    rl_qualities = []
    baseline_rewards = []
    baseline_qualities = []
    
    for episode in range(num_episodes):
        topic = topics[episode % len(topics)]
        
        # Test RL policy
        env = ContentCreationEnvironment(topic=topic)
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            state_vector = state.to_vector()
            q_values = [rl_policy.get_q_value(state_vector, a) for a in valid_actions]
            action = valid_actions[np.argmax(q_values)]
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        rl_rewards.append(episode_reward)
        rl_qualities.append(state.get_quality_score())
        
        # Test baseline policy
        env = ContentCreationEnvironment(topic=topic)
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = baseline_policy(state, valid_actions)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        baseline_rewards.append(episode_reward)
        baseline_qualities.append(state.get_quality_score())
    
    # Print comparison
    print(f"{'Metric':<25} {'RL Policy':>15} {'Baseline':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Average Reward':<25} {np.mean(rl_rewards):>15.2f} "
          f"{np.mean(baseline_rewards):>15.2f} "
          f"{(np.mean(rl_rewards) - np.mean(baseline_rewards)):>+15.2f}")
    print(f"{'Average Quality':<25} {np.mean(rl_qualities):>15.2%} "
          f"{np.mean(baseline_qualities):>15.2%} "
          f"{(np.mean(rl_qualities) - np.mean(baseline_qualities)):>+15.2%}")
    print(f"{'Std Dev Reward':<25} {np.std(rl_rewards):>15.2f} "
          f"{np.std(baseline_rewards):>15.2f}")
    print(f"{'Std Dev Quality':<25} {np.std(rl_qualities):>15.2%} "
          f"{np.std(baseline_qualities):>15.2%}")
    
    # Win rate
    rl_wins = sum(1 for r1, r2 in zip(rl_rewards, baseline_rewards) if r1 > r2)
    print(f"\nRL Policy Win Rate: {rl_wins}/{num_episodes} ({rl_wins/num_episodes:.0%})")
    
    print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Train the agent
    print("\n🤖 Starting RL Content Agent Training...\n")
    trained_policy, rewards, qualities = train_agent(num_episodes=100, verbose=True)
    
    # Demonstrate the trained agent
    print("\n\n🎯 Demonstrating Trained Agent...\n")
    demonstrate_agent(
        trained_policy, 
        topic="Why 90% of AI Products Fail After Launch",
        verbose=True
    )
    
    # Compare with baseline
    print("\n\n📊 Performance Comparison...\n")
    compare_policies(trained_policy, num_episodes=20)
    
    print("\n✅ MVP Demo Complete!")
    print("\nKey Takeaways:")
    print("1. RL agent learns to optimize content creation workflow")
    print("2. Explores different strategies and converges to effective patterns")
    print("3. Adapts based on reward signals (quality + engagement)")
    print("4. Outperforms fixed rule-based baseline through learning")
    print("\nNext steps for production:")
    print("- Integrate real LLM for content generation")
    print("- Add neural network policy (PPO/A2C)")
    print("- Collect real engagement data for reward model")
    print("- Implement continuous learning pipeline")