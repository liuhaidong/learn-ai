"""
RL-Based Content Optimization Agent with Real LLM Integration
Python Implementation with OpenAI-Compatible API

Requirements:
    pip install openai numpy

Usage:
    export OPENAI_API_KEY="your-api-key"
    export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI
    export OPENAI_MODEL="gpt-4"  # Optional, defaults to gpt-4
    
    python rl_content_agent.py

For other providers (e.g., Azure, local models):
    export OPENAI_BASE_URL="https://your-endpoint/v1"
    export OPENAI_MODEL="your-model-name"
"""

import os
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import time

# ============================================================================
# LLM CLIENT
# ============================================================================

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Error: openai package not installed.")
    print("Install with: pip install openai")
    exit(1)


class LLMClient:
    """Handles all LLM API calls using OpenAI-compatible API"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        """
        Initialize LLM client
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: API base URL (defaults to OPENAI_BASE_URL env var or OpenAI)
            model: Model name (defaults to OPENAI_MODEL env var or gpt-4)
        """
        self.api_key = "sk-rmoLqMBRKQ08WHiKikzRjEPEZipyCAviNkkNLTGGauhomeZX"#api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = "https://api.moonshot.cn/v1"#base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = "kimi-k2-turbo-preview"#model or os.environ.get("OPENAI_MODEL", "gpt-4")
        
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize client
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        print(f"✓ LLM Client initialized")
        print(f"  Model: {self.model}")
        if self.base_url:
            print(f"  Base URL: {self.base_url}")
    
    def call_llm(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        """
        Call LLM API with OpenAI-compatible interface
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
            
        Returns:
            str: The LLM response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract text from response
            text = response.choices[0].message.content
            return text.strip()
            
        except Exception as e:
            print(f"LLM API Error: {e}")
            raise


# ============================================================================
# STATE REPRESENTATION
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
    platform_fit: float = 0.0
    
    # Current stage
    current_stage: int = 0  # 0=Research, 1=Outline, 2=Draft, 3=Refine, 4=Done
    
    # Metadata
    revision_count: int = 0
    tokens_used: int = 0
    
    # Content storage
    research_data: List[str] = field(default_factory=list)
    outline: List[str] = field(default_factory=list)
    hook: str = ""
    sections: List[str] = field(default_factory=list)
    full_draft: str = ""
    
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
            float(self.current_stage) / 4.0,
            min(self.revision_count / 10.0, 1.0),
            min(self.tokens_used / 10000.0, 1.0),
        ])
    
    def get_quality_score(self) -> float:
        """Overall quality metric"""
        return np.mean([
            self.hook_strength,
            self.clarity_score,
            self.evidence_density,
            self.narrative_flow,
            self.platform_fit,
        ])


# ============================================================================
# ACTION SPACE
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
    target_section: int = 0
    intensity: float = 0.5
    
    def __repr__(self):
        return f"Action({self.action_type.name}, section={self.target_section}, intensity={self.intensity:.2f})"


# ============================================================================
# ENVIRONMENT WITH REAL LLM
# ============================================================================

class ContentCreationEnvironment:
    """Simulates content creation with real LLM calls"""
    
    def __init__(self, topic: str, platform: str = "wechat", 
                 max_steps: int = 15, llm_client: Optional[LLMClient] = None):
        self.topic = topic
        self.platform = platform
        self.max_steps = max_steps
        self.current_step = 0
        self.state = ContentState()
        self.target_sections = 5
        
        # LLM client
        if llm_client is None:
            raise ValueError("LLMClient required for ContentCreationEnvironment")
        self.llm = llm_client
    
    def reset(self) -> ContentState:
        """Reset environment to initial state"""
        self.current_step = 0
        self.state = ContentState()
        return self.state
    
    def _execute_research(self, intensity: float) -> Dict:
        """Execute research action using LLM"""
        
        num_findings = max(3, int(intensity * 5))
        prompt = f"""You are a research assistant. Research the topic: "{self.topic}"

Provide {num_findings} key research findings, statistics, or insights.
Focus on recent data, case studies, and actionable information.
Format as a bulleted list with each point starting with a dash (-).

Respond only with the research findings, nothing else."""
        
        research = self.llm.call_llm(prompt, max_tokens=600)
        
        # Parse research findings
        findings = [line.strip() for line in research.split('\n') 
                   if line.strip() and (line.strip().startswith('-') or 
                                       line.strip().startswith('•') or 
                                       line.strip().startswith('*'))]
        
        self.state.research_data.extend(findings)
        
        # Evaluate research quality
        word_count = len(research.split())
        import re
        has_numbers = bool(re.search(r'\d+%|\$\d+|\d+ [a-z]+', research, re.I))
        
        improvement = intensity * 0.3
        self.state.research_completeness = min(1.0, self.state.research_completeness + improvement)
        self.state.evidence_density = min(1.0, self.state.evidence_density + (0.3 if has_numbers else 0.1))
        
        return {
            'research': research,
            'improvement': improvement,
            'word_count': word_count,
            'findings_count': len(findings)
        }
    
    def _execute_create_outline(self, intensity: float) -> Dict:
        """Create outline using LLM"""
        
        research_context = '\n'.join(self.state.research_data[:10])
        
        prompt = f"""Create a detailed outline for a {self.platform} article titled: "{self.topic}"

Research context:
{research_context}

Create {self.target_sections} main sections with clear, compelling headers.
Each section should have 2-3 key points.

Format exactly as:
1. [Section Title]
   - Key point
   - Key point
2. [Section Title]
   - Key point
   - Key point

Respond only with the outline, nothing else."""
        
        outline = self.llm.call_llm(prompt, max_tokens=800)
        
        # Parse outline sections
        import re
        sections = re.split(r'\n(?=\d+\.)', outline)
        sections = [s.strip() for s in sections if s.strip()]
        self.state.outline = sections
        
        improvement = intensity * 0.4
        self.state.outline_completeness = min(1.0, self.state.outline_completeness + improvement)
        self.state.narrative_flow = min(1.0, self.state.narrative_flow + improvement * 0.6)
        
        return {
            'outline': outline,
            'sections_created': len(sections),
            'improvement': improvement
        }
    
    def _execute_write_hook(self, intensity: float) -> Dict:
        """Write hook using LLM"""
        
        research_context = '\n'.join(self.state.research_data[:5])
        
        platform_guidance = {
            'wechat': 'Professional tone suitable for Chinese business audience',
            'twitter': 'Casual, engaging Twitter style',
            'linkedin': 'Professional but conversational LinkedIn style'
        }
        
        prompt = f"""Write a compelling opening hook for a {self.platform} article: "{self.topic}"

Context:
{research_context}

Requirements:
- 2-3 sentences maximum
- Start with a shocking statistic, question, or bold statement
- Create curiosity and urgency
- {platform_guidance.get(self.platform, 'Engaging and professional tone')}

Write only the hook, nothing else."""
        
        hook = self.llm.call_llm(prompt, max_tokens=300, temperature=0.8)
        self.state.hook = hook
        
        # Evaluate hook quality
        has_question = '?' in hook
        import re
        has_numbers = bool(re.search(r'\d+%|\d+ [a-z]+', hook, re.I))
        word_count = len(hook.split())
        
        hook_quality = (0.3 if has_question else 0) + (0.4 if has_numbers else 0) + \
                      (0.3 if word_count < 50 else 0.1)
        
        improvement = intensity * 0.5
        self.state.hook_strength = min(1.0, self.state.hook_strength + improvement + hook_quality * 0.3)
        self.state.draft_completeness = min(1.0, self.state.draft_completeness + 0.15)
        
        return {
            'hook': hook,
            'hook_quality': hook_quality,
            'improvement': improvement
        }
    
    def _execute_write_section(self, section_idx: int, intensity: float) -> Dict:
        """Write a section using LLM"""
        
        outline_section = self.state.outline[section_idx] if section_idx < len(self.state.outline) else f"Section {section_idx + 1}"
        research_context = '\n'.join(self.state.research_data[:8])
        
        platform_style = {
            'wechat': 'Professional Chinese business style with clear structure',
            'twitter': 'Conversational, punchy style suitable for social media',
            'linkedin': 'Professional but approachable tone'
        }
        
        prompt = f"""Write content for this section of the article "{self.topic}":

Section outline:
{outline_section}

Research context:
{research_context}

Requirements:
- 150-200 words
- Include specific examples or data points
- Clear, engaging writing
- {platform_style.get(self.platform, 'Professional style')}
- Do not include the section title, only the content

Write the section content:"""
        
        section_content = self.llm.call_llm(prompt, max_tokens=600, temperature=0.7)
        
        # Ensure array is large enough
        while len(self.state.sections) <= section_idx:
            self.state.sections.append("")
        self.state.sections[section_idx] = section_content
        
        # Evaluate section quality
        word_count = len(section_content.split())
        import re
        has_examples = bool(re.search(r'example|instance|case|such as|for example', section_content, re.I))
        has_data = bool(re.search(r'\d+%|\$\d+|\d+ [a-z]+', section_content, re.I))
        
        section_quality = (0.3 if word_count >= 100 else 0.1) + \
                         (0.3 if has_examples else 0) + \
                         (0.4 if has_data else 0)
        
        completeness = len([s for s in self.state.sections if s]) / self.target_sections
        self.state.draft_completeness = completeness
        self.state.clarity_score = min(1.0, self.state.clarity_score + section_quality * 0.2)
        
        return {
            'section_content': section_content,
            'section_quality': section_quality,
            'word_count': word_count
        }
    
    def _execute_refine_content(self, intensity: float) -> Dict:
        """Refine content using LLM"""
        
        current_content = f"{self.state.hook}\n\n" + "\n\n".join([s for s in self.state.sections if s])
        
        # Limit content length for prompt
        content_preview = current_content[:2000]
        
        prompt = f"""Refine and improve this content for {self.platform}:

{content_preview}

Focus on:
- Clarity and readability
- Flow between sections
- Stronger transitions
- Platform-specific optimization ({self.platform})
- Grammar and polish

Provide the complete refined version with all improvements:"""
        
        refined = self.llm.call_llm(prompt, max_tokens=1200, temperature=0.5)
        
        # Update sections with refined content
        refined_sections = refined.split('\n\n')
        refined_sections = [s.strip() for s in refined_sections if len(s.strip()) > 50]
        
        if refined_sections:
            self.state.hook = refined_sections[0]
            self.state.sections = refined_sections[1:] if len(refined_sections) > 1 else self.state.sections
        
        improvement = intensity * 0.2
        self.state.clarity_score = min(1.0, self.state.clarity_score + improvement)
        self.state.narrative_flow = min(1.0, self.state.narrative_flow + improvement)
        self.state.platform_fit = min(1.0, self.state.platform_fit + improvement * 0.8)
        self.state.revision_count += 1
        
        return {'improvement': improvement}
    
    def step(self, action: Action) -> Tuple[ContentState, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)"""
        
        self.current_step += 1
        reward = 0.0
        info = {}
        
        try:
            if action.action_type == ActionType.RESEARCH:
                result = self._execute_research(action.intensity)
                reward = 0.1 * result['improvement']
                info['action_result'] = f"Researched: Found {result['findings_count']} insights"
                self.state.tokens_used += int(result['word_count'] * 1.3)
                
            elif action.action_type == ActionType.CREATE_OUTLINE:
                if self.state.research_completeness < 0.3:
                    reward = -0.2
                    info['action_result'] = "Failed: Need more research first"
                else:
                    result = self._execute_create_outline(action.intensity)
                    reward = 0.15 * result['improvement']
                    info['action_result'] = f"Created outline: {result['sections_created']} sections"
                    self.state.current_stage = max(self.state.current_stage, 1)
                    self.state.tokens_used += 500
                    
            elif action.action_type == ActionType.WRITE_HOOK:
                if self.state.outline_completeness < 0.4:
                    reward = -0.2
                    info['action_result'] = "Failed: Need outline first"
                else:
                    result = self._execute_write_hook(action.intensity)
                    reward = 0.2 * result['improvement'] + result['hook_quality'] * 0.1
                    info['action_result'] = f"Wrote hook (quality: {result['hook_quality']*100:.0f}%)"
                    self.state.current_stage = max(self.state.current_stage, 2)
                    self.state.tokens_used += 200
                    
            elif action.action_type == ActionType.WRITE_SECTION:
                if self.state.outline_completeness < 0.5:
                    reward = -0.15
                    info['action_result'] = "Failed: Outline not ready"
                else:
                    result = self._execute_write_section(action.target_section, action.intensity)
                    reward = 0.15 + result['section_quality'] * 0.1
                    info['action_result'] = f"Wrote section {action.target_section + 1} ({result['word_count']} words, quality: {result['section_quality']*100:.0f}%)"
                    self.state.current_stage = max(self.state.current_stage, 2)
                    self.state.tokens_used += int(result['word_count'] * 1.3)
                    
            elif action.action_type == ActionType.REFINE_CONTENT:
                if self.state.draft_completeness < 0.4:
                    reward = -0.1
                    info['action_result'] = "Failed: Not enough content to refine"
                else:
                    result = self._execute_refine_content(action.intensity)
                    reward = 0.1 * result['improvement']
                    info['action_result'] = f"Refined content (revision {self.state.revision_count})"
                    self.state.current_stage = max(self.state.current_stage, 3)
                    self.state.tokens_used += 800
                    
            elif action.action_type == ActionType.FINALIZE:
                # Compile full draft
                self.state.full_draft = f"{self.state.hook}\n\n" + "\n\n".join([s for s in self.state.sections if s])
                print('*'*30)
                print(self.state.full_draft)
                quality = self.state.get_quality_score()
                completeness = (self.state.research_completeness + 
                               self.state.outline_completeness + 
                               self.state.draft_completeness) / 3.0
                
                if completeness > 0.7 and quality > 0.6:
                    reward = 1.0 + quality
                    info['action_result'] = f"SUCCESS! Quality: {quality*100:.0f}%"
                elif completeness > 0.5:
                    reward = 0.5
                    info['action_result'] = f"Completed (Quality: {quality*100:.0f}%)"
                else:
                    reward = -0.5
                    info['action_result'] = f"Incomplete ({completeness*100:.0f}% done)"
                
                self.state.current_stage = 4
                
        except Exception as e:
            reward = -0.3
            info['action_result'] = f"Error: {str(e)}"
            print(f"Action execution error: {e}")
        
        # Token budget penalty
        if self.state.tokens_used > 8000:
            reward -= 0.1
        
        # Check if done
        done = (self.state.current_stage == 4 or self.current_step >= self.max_steps)
        
        if done and self.state.current_stage != 4:
            reward -= 0.3
            info['action_result'] = "TIMEOUT: Max steps reached"
        
        return self.state, reward, done, info
    
    def get_valid_actions(self) -> List[Action]:
        """Return list of valid actions in current state"""
        actions = []
        
        # Research actions
        for intensity in [0.3, 0.5, 0.8]:
            actions.append(Action(ActionType.RESEARCH, intensity=intensity))
        
        # Outline actions
        if self.state.research_completeness > 0.2:
            for intensity in [0.5, 0.8]:
                actions.append(Action(ActionType.CREATE_OUTLINE, intensity=intensity))
        
        # Hook actions
        if self.state.outline_completeness > 0.3:
            for intensity in [0.5, 0.8]:
                actions.append(Action(ActionType.WRITE_HOOK, intensity=intensity))
        
        # Section actions
        if self.state.outline_completeness > 0.4:
            for i in range(self.target_sections):
                actions.append(Action(ActionType.WRITE_SECTION, target_section=i, intensity=0.7))
        
        # Refine actions
        if self.state.draft_completeness > 0.3:
            actions.append(Action(ActionType.REFINE_CONTENT, intensity=0.6))
        
        # Finalize
        actions.append(Action(ActionType.FINALIZE, intensity=1.0))
        
        return actions


# ============================================================================
# SIMPLE Q-LEARNING POLICY
# ============================================================================

class SimplePolicy:
    """Q-learning policy using linear function approximation"""
    
    def __init__(self, state_dim: int, learning_rate: float = 0.01):
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = 0.95
        self.epsilon = 0.5
        
        # Q-value weights for each action type
        self.weights = {
            action_type: np.random.randn(state_dim) * 0.01
            for action_type in ActionType
        }
        
        self.episodes_trained = 0
    
    def get_q_value(self, state_vector: np.ndarray, action: Action) -> float:
        """Compute Q(s, a)"""
        weights = self.weights[action.action_type]
        q_value = np.dot(state_vector, weights) * action.intensity
        return q_value
    
    def select_action(self, state: ContentState, valid_actions: List[Action]) -> Action:
        """Epsilon-greedy action selection"""
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            state_vector = state.to_vector()
            q_values = [self.get_q_value(state_vector, a) for a in valid_actions]
            best_idx = np.argmax(q_values)
            return valid_actions[best_idx]
    
    def update(self, state: ContentState, action: Action, reward: float,
               next_state: ContentState, valid_next_actions: List[Action], done: bool):
        """Q-learning update"""
        
        state_vector = state.to_vector()
        current_q = self.get_q_value(state_vector, action)
        
        if done:
            target_q = reward
        else:
            next_state_vector = next_state.to_vector()
            next_q_values = [self.get_q_value(next_state_vector, a) for a in valid_next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            target_q = reward + self.gamma * max_next_q
        
        td_error = target_q - current_q
        
        # Gradient update
        gradient = state_vector * action.intensity
        self.weights[action.action_type] += self.learning_rate * td_error * gradient
    
    def decay_exploration(self):
        """Reduce exploration over time"""
        self.epsilon = max(0.05, self.epsilon * 0.99)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_agent(llm_client: LLMClient, num_episodes: int = 10, verbose: bool = True) -> Tuple[SimplePolicy, List, List]:
    """Train the RL agent"""
    
    topics = [
        "Why 90% of AI Products Fail After Launch",
        "The Future of Remote Work in 2025",
        "Cybersecurity Best Practices for Startups",
        "Machine Learning for Small Businesses",
        "Sustainable Tech Trends",
    ]
    
    print("=" * 70)
    print("TRAINING RL CONTENT OPTIMIZATION AGENT WITH REAL LLM")
    print("=" * 70)
    print()
    
    # Initialize environment and policy
    env = ContentCreationEnvironment(topics[0], llm_client=llm_client)
    state_dim = len(env.state.to_vector())
    policy = SimplePolicy(state_dim=state_dim)
    
    episode_rewards = []
    episode_qualities = []
    
    for episode in range(num_episodes):
        topic = random.choice(topics)
        env = ContentCreationEnvironment(topic, platform="wechat", max_steps=12, llm_client=llm_client)
        
        state = env.reset()
        episode_reward = 0
        done = False
        
        if verbose:
            print(f"📝 Episode {episode + 1}/{num_episodes}: \"{topic}\"")
        
        step_count = 0
        while not done:
            step_count += 1
            valid_actions = env.get_valid_actions()
            action = policy.select_action(state, valid_actions)
            
            next_state, reward, done, info = env.step(action)
            
            next_valid_actions = env.get_valid_actions() if not done else []
            policy.update(state, action, reward, next_state, next_valid_actions, done)
            
            episode_reward += reward
            state = next_state
        
        policy.decay_exploration()
        policy.episodes_trained += 1
        
        episode_rewards.append(episode_reward)
        final_quality = state.get_quality_score()
        episode_qualities.append(final_quality)
        
        if verbose:
            print(f"   ✓ Reward: {episode_reward:6.2f} | Quality: {final_quality*100:5.0f}% | "
                  f"Steps: {step_count:2d} | ε: {policy.epsilon:.3f}")
            print()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    avg_reward = np.mean(episode_rewards[-min(5, num_episodes):])
    avg_quality = np.mean(episode_qualities[-min(5, num_episodes):])
    
    print(f"Final average reward:  {avg_reward:.2f}")
    print(f"Final average quality: {avg_quality*100:.0f}%")
    print()
    
    return policy, episode_rewards, episode_qualities


# ============================================================================
# DEMONSTRATION FUNCTION
# ============================================================================

def demonstrate_agent(llm_client: LLMClient, policy: Optional[SimplePolicy] = None, 
                     topic: str = None, verbose: bool = True):
    """Demonstrate trained agent creating content"""
    
    if topic is None:
        topic = "Why 90% of AI Products Fail After Launch"
    
    print("=" * 70)
    print(f"DEMONSTRATION: Creating Content")
    print("=" * 70)
    print(f"Topic: {topic}")
    print()
    
    # Quick training if no policy provided
    if policy is None:
        print("Training agent (5 episodes)...")
        policy, _, _ = train_agent(llm_client, num_episodes=5, verbose=False)
        print("✓ Training complete\n")
    
    # Create environment
    env = ContentCreationEnvironment(topic, platform="wechat", max_steps=15, llm_client=llm_client)
    
    state = env.reset()
    step = 0
    done = False
    total_reward = 0
    
    # Greedy policy (no exploration)
    original_epsilon = policy.epsilon
    policy.epsilon = 0
    
    print("Creating content step by step...\n")
    
    while not done:
        step += 1
        valid_actions = env.get_valid_actions()
        
        # Greedy selection
        state_vector = state.to_vector()
        q_values = [policy.get_q_value(state_vector, a) for a in valid_actions]
        best_action = valid_actions[np.argmax(q_values)]
        
        next_state, reward, done, info = env.step(best_action)
        total_reward += reward
        
        if verbose:
            print(f"Step {step:2d}: {best_action.action_type.name}")
            print(f"         → {info['action_result']}")
            print(f"         → Reward: {reward:+.2f} | Cumulative: {total_reward:+.2f}")
            print(f"         → Quality: {state.get_quality_score()*100:.0f}% | "
                  f"Draft: {state.draft_completeness*100:.0f}% | Tokens: {state.tokens_used}")
            print()
        
        state = next_state
    
    # Restore epsilon
    policy.epsilon = original_epsilon
    
    # Final summary
    print("=" * 70)
    print("FINAL CONTENT STATE")
    print("=" * 70)
    print(f"Research Completeness: {state.research_completeness*100:5.0f}%")
    print(f"Outline Completeness:  {state.outline_completeness*100:5.0f}%")
    print(f"Draft Completeness:    {state.draft_completeness*100:5.0f}%")
    print(f"Hook Strength:         {state.hook_strength*100:5.0f}%")
    print(f"Clarity Score:         {state.clarity_score*100:5.0f}%")
    print(f"Evidence Density:      {state.evidence_density*100:5.0f}%")
    print(f"Narrative Flow:        {state.narrative_flow*100:5.0f}%")
    print(f"Platform Fit:          {state.platform_fit*100:5.0f}%")
    print()
    print(f"Overall Quality:       {state.get_quality_score()*100:5.0f}%")
    print(f"Total Reward:          {total_reward:+.2f}")
    print(f"Tokens Used:           {state.tokens_used}")
    print(f"Revisions:             {state.revision_count}")
    print()
    
    # Display generated content
    print("=" * 70)
    print("GENERATED CONTENT")
    print("=" * 70)
    print()
    
    if state.hook:
        print("HOOK:")
        print("-" * 70)
        print(state.hook)
        print()
    
    if state.sections:
        print("SECTIONS:")
        print("-" * 70)
        for i, section in enumerate(state.sections):
            if section:
                print(f"\n[Section {i+1}]")
                print(section)
                print()
    
    if state.full_draft:
        print("=" * 70)
        print("FULL ARTICLE")
        print("=" * 70)
        print()
        print(state.full_draft)
        print()
    
    print("=" * 70)
    
    return state, total_reward


# ============================================================================
# COMPARISON FUNCTION
# ============================================================================

def compare_policies(llm_client: LLMClient, rl_policy: SimplePolicy, num_episodes: int = 10):
    """Compare RL policy vs baseline"""
    
    print("=" * 70)
    print("COMPARING RL POLICY VS BASELINE")
    print("=" * 70)
    print()
    
    topics = [
        "Why 90% of AI Products Fail After Launch",
        "The Future of Remote Work in 2025",
        "Sustainable Tech: Green Computing Trends",
    ]
    
    rl_rewards = []
    rl_qualities = []
    baseline_rewards = []
    baseline_qualities = []
    
    def baseline_policy(state: ContentState, valid_actions: List[Action]) -> Action:
        """Simple heuristic baseline"""
        if state.research_completeness < 0.8:
            research_actions = [a for a in valid_actions if a.action_type == ActionType.RESEARCH]
            if research_actions:
                return max(research_actions, key=lambda a: a.intensity)
        
        if state.outline_completeness < 0.8:
            outline_actions = [a for a in valid_actions if a.action_type == ActionType.CREATE_OUTLINE]
            if outline_actions:
                return max(outline_actions, key=lambda a: a.intensity)
        
        if state.hook_strength < 0.7:
            hook_actions = [a for a in valid_actions if a.action_type == ActionType.WRITE_HOOK]
            if hook_actions:
                return max(hook_actions, key=lambda a: a.intensity)
        
        if state.draft_completeness < 0.9:
            section_actions = [a for a in valid_actions if a.action_type == ActionType.WRITE_SECTION]
            if section_actions:
                return section_actions[0]
        
        if state.revision_count < 2:
            refine_actions = [a for a in valid_actions if a.action_type == ActionType.REFINE_CONTENT]
            if refine_actions:
                return refine_actions[0]
        
        finalize_actions = [a for a in valid_actions if a.action_type == ActionType.FINALIZE]
        return finalize_actions[0] if finalize_actions else valid_actions[0]
    
    for episode in range(num_episodes):
        topic = topics[episode % len(topics)]
        
        # Test RL policy
        env = ContentCreationEnvironment(topic, llm_client=llm_client)
        state = env.reset()
        episode_reward = 0
        done = False
        
        rl_policy.epsilon = 0  # Greedy
        
        while not done:
            valid_actions = env.get_valid_actions()
            state_vector = state.to_vector()
            q_values = [rl_policy.get_q_value(state_vector, a) for a in valid_actions]
            action = valid_actions[np.argmax(q_values)]
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        rl_rewards.append(episode_reward)
        rl_qualities.append(state.get_quality_score())
        
        # Test baseline
        env = ContentCreationEnvironment(topic, llm_client=llm_client)
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
        
        print(f"Episode {episode+1}/{num_episodes} - RL: {rl_rewards[-1]:.2f} | Baseline: {baseline_rewards[-1]:.2f}")
    
    # Print comparison
    print()
    print(f"{'Metric':<25} {'RL Policy':>15} {'Baseline':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Average Reward':<25} {np.mean(rl_rewards):>15.2f} "
          f"{np.mean(baseline_rewards):>15.2f} "
          f"{(np.mean(rl_rewards) - np.mean(baseline_rewards)):>+15.2f}")
    print(f"{'Average Quality':<25} {np.mean(rl_qualities)*100:>14.0f}% "
          f"{np.mean(baseline_qualities)*100:>14.0f}% "
          f"{(np.mean(rl_qualities) - np.mean(baseline_qualities))*100:>+14.0f}%")
    print(f"{'Std Dev Reward':<25} {np.std(rl_rewards):>15.2f} "
          f"{np.std(baseline_rewards):>15.2f}")
    
    rl_wins = sum(1 for r1, r2 in zip(rl_rewards, baseline_rewards) if r1 > r2)
    print()
    print(f"RL Policy Win Rate: {rl_wins}/{num_episodes} ({rl_wins/num_episodes*100:.0f}%)")
    print("=" * 70)
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("RL CONTENT OPTIMIZATION AGENT WITH REAL LLM INTEGRATION")
    print("OpenAI-Compatible API")
    print("="*70 + "\n")
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    
    # Initialize LLM client
    try:
        llm_client = LLMClient()
        print()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease set your API credentials:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\nOptional configuration:")
        print("  export OPENAI_BASE_URL='https://api.openai.com/v1'  # Default")
        print("  export OPENAI_MODEL='gpt-4'  # Default")
        print("\nFor other providers:")
        print("  export OPENAI_BASE_URL='https://your-custom-endpoint/v1'")
        print("  export OPENAI_MODEL='your-model-name'")
        return
    
    # Menu
    while True:
        print("\nSelect an option:")
        print("1. Train agent (10 episodes)")
        print("2. Quick demo (train + demonstrate)")
        print("3. Compare RL vs Baseline (requires trained agent)")
        print("4. Custom demo with your topic")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\n")
            policy, rewards, qualities = train_agent(llm_client, num_episodes=10, verbose=True)
            
            # Option to save policy
            save = input("\nSave trained policy? (y/n): ").strip().lower()
            if save == 'y':
                import pickle
                with open('trained_policy.pkl', 'wb') as f:
                    pickle.dump(policy, f)
                print("✓ Policy saved to trained_policy.pkl")
        
        elif choice == "2":
            print("\n")
            policy, _, _ = train_agent(llm_client, num_episodes=5, verbose=True)
            print("\n")
            time.sleep(1)
            demonstrate_agent(llm_client, policy, verbose=True)
        
        elif choice == "3":
            # Try to load saved policy
            try:
                import pickle
                with open('trained_policy.pkl', 'rb') as f:
                    policy = pickle.load(f)
                print("✓ Loaded saved policy\n")
            except:
                print("No saved policy found. Training new one...\n")
                policy, _, _ = train_agent(llm_client, num_episodes=10, verbose=False)
            
            compare_policies(llm_client, policy, num_episodes=10)
        
        elif choice == "4":
            topic = input("\nEnter your topic: ").strip()
            if not topic:
                print("Topic cannot be empty.")
                continue
            
            # Try to load saved policy
            try:
                import pickle
                with open('trained_policy.pkl', 'rb') as f:
                    policy = pickle.load(f)
                print("✓ Loaded saved policy\n")
            except:
                print("No saved policy found. Training new one...\n")
                policy, _, _ = train_agent(llm_client, num_episodes=5, verbose=False)
                print()
            
            demonstrate_agent(llm_client, policy, topic=topic, verbose=True)
        
        elif choice == "5":
            print("\n👋 Goodbye!\n")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()