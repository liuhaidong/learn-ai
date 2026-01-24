# pip install gymnasium numpy stable-baselines3 torch anthropic openai

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
import hashlib
from datetime import datetime
import asyncio


# ============================================================================
# 1. STRUCTURED STATE (same as before)
# ============================================================================

@dataclass
class ContentSemantics:
    claim_density: float
    evidence_to_claim_ratio: float
    abstractness_score: float
    redundancy_score: float
    concrete_example_count: int
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.claim_density,
            self.evidence_to_claim_ratio,
            self.abstractness_score,
            self.redundancy_score,
            self.concrete_example_count / 10.0
        ], dtype=np.float32)


@dataclass
class RhetoricalQuality:
    belief_conflict_strength: float
    narrative_tension: float
    hook_strength: float
    logical_coherence: float
    cognitive_load: float
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.belief_conflict_strength,
            self.narrative_tension,
            self.hook_strength,
            self.logical_coherence,
            self.cognitive_load
        ], dtype=np.float32)


class Platform(Enum):
    TWITTER = 0
    WECHAT = 1
    BLOG = 2


@dataclass
class PlatformContext:
    platform: Platform
    reader_sophistication: float
    topic_fatigue: float
    style_conformity: float
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.platform.value / 2.0,
            self.reader_sophistication,
            self.topic_fatigue,
            self.style_conformity
        ], dtype=np.float32)


# ============================================================================
# 2. EDITORIAL ACTIONS
# ============================================================================

class EditorialAction(Enum):
    ADD_COUNTER_ARGUMENT = 0
    MOVE_EXAMPLE_EARLIER = 1
    CUT_OPENING = 2
    SHARPEN_CLAIM = 3
    ADD_CONCRETE_CASE = 4
    ADD_DATA_POINT = 5
    INCREASE_CONTRAST = 6
    REDUCE_HEDGING = 7
    FINALIZE = 8


# ============================================================================
# 3. LLM CANDIDATE GENERATOR
# ============================================================================

class LLMCandidateGenerator:
    """
    Generates multiple edit candidates using Claude API
    Option B: LLM generates alternatives, RL selects best
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.action_prompts = {
            EditorialAction.ADD_COUNTER_ARGUMENT: """
Current text: {text}

Task: Add a counter-argument to strengthen the piece.
Requirements:
- Introduce a credible opposing viewpoint
- Acknowledge its merit before addressing it
- Use phrases like "However, critics argue..." or "Some researchers suggest..."
- Keep it 2-3 sentences
- Maintain professional tone

Generate 3 different ways to add this counter-argument.""",

            EditorialAction.ADD_CONCRETE_CASE: """
Current text: {text}

Task: Add a concrete example or case study.
Requirements:
- Replace abstract claims with specific instances
- Use real or realistic examples
- Include numbers, names, or specific details
- Keep it 2-3 sentences
- Make it memorable

Generate 3 different concrete examples to add.""",

            EditorialAction.SHARPEN_CLAIM: """
Current text: {text}

Task: Sharpen the main claims to be more direct and assertive.
Requirements:
- Remove hedging words (might, could, perhaps)
- Make claims more definitive where evidence supports it
- Strengthen weak verbs
- Maintain accuracy - don't overclaim
- Keep the same meaning, just clearer

Generate 3 versions with sharpened claims.""",

            EditorialAction.CUT_OPENING: """
Current text: {text}

Task: Tighten the opening by removing redundancy.
Requirements:
- Cut the first 1-2 sentences if they're generic
- Start with the hook or main point
- Remove throat-clearing phrases
- Keep essential context
- Make it punchier

Generate 3 versions with different opening cuts.""",
        }
    
    async def generate_candidates(
        self, 
        text: str, 
        action: EditorialAction,
        n_candidates: int = 3
    ) -> List[str]:
        """
        Generate multiple edit candidates for given action
        In production: calls Claude API
        For demo: returns synthetic candidates
        """
        
        if action not in self.action_prompts:
            return [text]  # No-op for unsupported actions
        
        # PRODUCTION: Uncomment to use real Claude API
        # candidates = await self._call_claude_api(text, action, n_candidates)
        
        # DEMO: Synthetic candidates
        candidates = self._generate_synthetic_candidates(text, action, n_candidates)
        
        return candidates
    
    async def _call_claude_api(self, text: str, action: EditorialAction, n: int) -> List[str]:
        """Real Claude API call"""
        # Example using Anthropic SDK
        """
        from anthropic import AsyncAnthropic
        
        client = AsyncAnthropic(api_key=self.api_key)
        
        prompt = self.action_prompts[action].format(text=text)
        
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response to extract n candidates
        response_text = message.content[0].text
        candidates = self._parse_candidates(response_text)
        return candidates[:n]
        """
        pass
    
    def _generate_synthetic_candidates(
        self, 
        text: str, 
        action: EditorialAction, 
        n: int
    ) -> List[str]:
        """Synthetic candidates for demo purposes"""
        
        candidates = []
        
        if action == EditorialAction.ADD_COUNTER_ARGUMENT:
            candidates = [
                text + "\n\nHowever, critics argue that this approach overlooks practical constraints. Implementation costs and organizational resistance remain significant barriers.",
                text + "\n\nYet skeptics point to conflicting evidence. Recent studies suggest the benefits may be overstated in certain contexts.",
                text + "\n\nStill, some researchers question these findings. Alternative methodologies have produced different results worth considering."
            ]
        
        elif action == EditorialAction.ADD_CONCRETE_CASE:
            candidates = [
                text + "\n\nConsider Microsoft's 2023 deployment: they reduced processing time by 47% within 8 weeks, serving 2.3M users.",
                text + "\n\nTake the case of a mid-sized retailer in Austin. After implementation, customer satisfaction scores jumped from 3.2 to 4.7 out of 5.",
                text + "\n\nLook at Denmark's national rollout in Q2 2024. With a €12M investment, they achieved 89% adoption among target users."
            ]
        
        elif action == EditorialAction.SHARPEN_CLAIM:
            sharpened = text.replace("might", "will").replace("could", "does")
            sharpened = sharpened.replace("perhaps", "").replace("possibly", "")
            candidates = [
                sharpened,
                sharpened.replace("suggests that", "proves that"),
                sharpened.replace("indicates", "demonstrates")
            ]
        
        elif action == EditorialAction.CUT_OPENING:
            sentences = text.split('.')
            candidates = [
                '.'.join(sentences[1:]) if len(sentences) > 1 else text,
                '.'.join(sentences[2:]) if len(sentences) > 2 else text,
                text.split('\n', 1)[1] if '\n' in text else text
            ]
        
        else:
            candidates = [text] * n
        
        return candidates[:n]


# ============================================================================
# 4. VALUE FUNCTION: RL Agent Selects Best Candidate
# ============================================================================

class CandidateValueNetwork:
    """
    Learned value function that scores candidate edits
    This is the RL component that learns from human feedback
    """
    
    def __init__(self, state_dim: int = 19):
        self.state_dim = state_dim
        # In production: neural network
        # For demo: simple heuristic-based scoring
        self.weights = {
            'claim_density': 0.3,
            'evidence_ratio': 0.25,
            'coherence': 0.2,
            'hook_strength': 0.15,
            'belief_conflict': 0.1
        }
    
    def score_candidate(
        self, 
        state_before: np.ndarray,
        state_after: np.ndarray,
        action: EditorialAction
    ) -> float:
        """
        Score how good a candidate edit is
        Higher score = better edit according to learned preferences
        """
        
        # Extract key features from states
        delta = state_after - state_before
        
        # Compute value based on learned preferences
        score = 0.0
        
        # Reward claim density increase (up to a point)
        claim_delta = delta[0]
        score += self.weights['claim_density'] * np.clip(claim_delta, -0.2, 0.2)
        
        # Reward evidence improvement
        evidence_delta = delta[1]
        score += self.weights['evidence_ratio'] * evidence_delta
        
        # Reward coherence
        coherence_delta = delta[8]  # logical_coherence index
        score += self.weights['coherence'] * coherence_delta
        
        # Reward hook strength
        hook_delta = delta[7]
        score += self.weights['hook_strength'] * hook_delta
        
        # Penalty for excessive changes
        total_change = np.abs(delta).sum()
        if total_change > 1.0:
            score -= 0.2
        
        return float(score)
    
    def select_best_candidate(
        self,
        state_before: np.ndarray,
        candidates: List[Tuple[str, np.ndarray]],
        action: EditorialAction
    ) -> Tuple[int, float]:
        """
        Select best candidate from LLM outputs
        Returns: (best_index, confidence_score)
        """
        
        scores = [
            self.score_candidate(state_before, state_after, action)
            for _, state_after in candidates
        ]
        
        best_idx = int(np.argmax(scores))
        confidence = float(scores[best_idx])
        
        return best_idx, confidence


# ============================================================================
# 5. HUMAN-IN-THE-LOOP WORKFLOW
# ============================================================================

@dataclass
class EditProposal:
    """A single edit suggestion from the agent"""
    original_text: str
    edited_text: str
    action: EditorialAction
    rationale: str
    confidence: float
    state_before: np.ndarray
    state_after: np.ndarray
    timestamp: str
    proposal_id: str
    
    def to_dict(self) -> Dict:
        return {
            'proposal_id': self.proposal_id,
            'original_text': self.original_text,
            'edited_text': self.edited_text,
            'action': self.action.name,
            'rationale': self.rationale,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


@dataclass
class HumanFeedback:
    """Human editor's response to a proposal"""
    proposal_id: str
    accepted: bool
    human_edit: Optional[str]  # If they made their own edit instead
    comments: str
    editor_id: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class HITLWorkflow:
    """
    Human-in-the-Loop workflow for shadow mode and assisted editing
    Logs all decisions for offline training
    """
    
    def __init__(self, storage_path: str = "hitl_data"):
        self.storage_path = storage_path
        self.proposals_log: List[EditProposal] = []
        self.feedback_log: List[HumanFeedback] = []
        self.agreement_rate: float = 0.0
    
    def suggest_edits(
        self,
        text: str,
        state: np.ndarray,
        candidates: List[Tuple[str, np.ndarray, EditorialAction, float]],
        top_k: int = 3
    ) -> List[EditProposal]:
        """
        Agent proposes top-k edits
        Human approves/rejects
        """
        
        # Sort by confidence
        sorted_candidates = sorted(candidates, key=lambda x: x[3], reverse=True)
        
        proposals = []
        for edited_text, state_after, action, confidence in sorted_candidates[:top_k]:
            proposal = EditProposal(
                original_text=text,
                edited_text=edited_text,
                action=action,
                rationale=self._generate_rationale(action, state, state_after),
                confidence=confidence,
                state_before=state,
                state_after=state_after,
                timestamp=datetime.now().isoformat(),
                proposal_id=self._generate_id(text, action)
            )
            proposals.append(proposal)
            self.proposals_log.append(proposal)
        
        return proposals
    
    def record_feedback(self, feedback: HumanFeedback):
        """Record human editor's decision"""
        self.feedback_log.append(feedback)
        self._update_metrics()
        self._save_to_disk()
    
    def shadow_mode(
        self,
        text: str,
        agent_edit: str,
        action: EditorialAction,
        human_edit: str,
        editor_id: str
    ):
        """
        Shadow mode: Agent suggests silently, human edits normally
        Log disagreements for later training
        """
        
        proposal = EditProposal(
            original_text=text,
            edited_text=agent_edit,
            action=action,
            rationale="Shadow mode suggestion",
            confidence=0.0,
            state_before=np.zeros(19),
            state_after=np.zeros(19),
            timestamp=datetime.now().isoformat(),
            proposal_id=self._generate_id(text, action)
        )
        
        # Check if agent suggestion matches human decision
        accepted = (agent_edit.strip() == human_edit.strip())
        
        feedback = HumanFeedback(
            proposal_id=proposal.proposal_id,
            accepted=accepted,
            human_edit=human_edit if not accepted else None,
            comments="Shadow mode observation",
            editor_id=editor_id,
            timestamp=datetime.now().isoformat()
        )
        
        self.proposals_log.append(proposal)
        self.feedback_log.append(feedback)
        self._update_metrics()
    
    def get_training_data(self) -> List[Dict]:
        """
        Export data for offline imitation learning
        Returns: List of (state, action, human_preference) tuples
        """
        
        training_data = []
        
        for proposal in self.proposals_log:
            feedback = next(
                (f for f in self.feedback_log if f.proposal_id == proposal.proposal_id),
                None
            )
            
            if feedback:
                training_data.append({
                    'state_before': proposal.state_before.tolist(),
                    'state_after': proposal.state_after.tolist(),
                    'action': proposal.action.name,
                    'agent_text': proposal.edited_text,
                    'human_accepted': feedback.accepted,
                    'human_text': feedback.human_edit,
                    'timestamp': proposal.timestamp
                })
        
        return training_data
    
    def _generate_rationale(
        self, 
        action: EditorialAction, 
        state_before: np.ndarray,
        state_after: np.ndarray
    ) -> str:
        """Generate human-readable rationale"""
        
        rationales = {
            EditorialAction.ADD_COUNTER_ARGUMENT:
                f"Low belief conflict ({state_before[5]:.2f}) → adding counterpoint increases engagement",
            EditorialAction.ADD_CONCRETE_CASE:
                f"High abstractness ({state_before[2]:.2f}) → concrete example improves clarity",
            EditorialAction.SHARPEN_CLAIM:
                f"Weak claim density ({state_before[0]:.2f}) → sharper language strengthens argument",
            EditorialAction.CUT_OPENING:
                f"High redundancy ({state_before[3]:.2f}) → tighter opening improves hook"
        }
        
        return rationales.get(action, f"Applying {action.name} to improve quality")
    
    def _generate_id(self, text: str, action: EditorialAction) -> str:
        """Generate unique ID for proposal"""
        content = f"{text[:50]}{action.name}{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _update_metrics(self):
        """Update tracking metrics"""
        if len(self.feedback_log) > 0:
            accepted = sum(1 for f in self.feedback_log if f.accepted)
            self.agreement_rate = accepted / len(self.feedback_log)
    
    def _save_to_disk(self):
        """Save logs for later analysis"""
        # In production: save to database
        data = {
            'proposals': [p.to_dict() for p in self.proposals_log],
            'feedback': [f.to_dict() for f in self.feedback_log],
            'metrics': {
                'agreement_rate': self.agreement_rate,
                'total_proposals': len(self.proposals_log)
            }
        }
        
        # Would save to file/database
        # with open(f"{self.storage_path}/hitl_log.json", 'w') as f:
        #     json.dump(data, f, indent=2)


# ============================================================================
# 6. IMITATION LEARNING: Bootstrap from Human Edits
# ============================================================================

class ImitationLearner:
    """
    Phase 1: Learn from human demonstrations
    Before RL, train policy to imitate good editors
    """
    
    def __init__(self):
        self.demonstrations: List[Dict] = []
    
    def add_demonstration(
        self,
        state_before: np.ndarray,
        action: EditorialAction,
        state_after: np.ndarray,
        human_approved: bool
    ):
        """Add a human editing demonstration"""
        self.demonstrations.append({
            'state': state_before,
            'action': action.value,
            'next_state': state_after,
            'weight': 1.0 if human_approved else 0.3
        })
    
    def train_policy(self, base_model: PPO, epochs: int = 10):
        """
        Supervised learning phase: train policy to match human edits
        """
        
        print(f"Imitation Learning: {len(self.demonstrations)} demonstrations")
        
        # Convert to training format
        states = np.array([d['state'] for d in self.demonstrations])
        actions = np.array([d['action'] for d in self.demonstrations])
        weights = np.array([d['weight'] for d in self.demonstrations])
        
        # In production: behavioral cloning with weighted samples
        # For demo: just print statistics
        
        print(f"  Most common action: {max(set(actions), key=list(actions).count)}")
        print(f"  Average weight: {weights.mean():.2f}")
        print(f"  Ready for RL fine-tuning")
        
        return base_model


# ============================================================================
# 7. INTEGRATED PRODUCTION SYSTEM
# ============================================================================

class ProductionEditorialSystem:
    """
    End-to-end system:
    1. LLM generates candidates
    2. RL agent selects best
    3. HITL workflow collects feedback
    4. Imitation learning bootstraps
    5. RL fine-tunes
    """
    
    def __init__(self, phase: str = "shadow"):
        self.llm_generator = LLMCandidateGenerator()
        self.value_network = CandidateValueNetwork()
        self.hitl = HITLWorkflow()
        self.imitation_learner = ImitationLearner()
        self.phase = phase  # "shadow", "assisted", or "autopilot"
        
        self.analyzer = ContentAnalyzer()
    
    async def process_content(
        self,
        text: str,
        editor_id: str = "demo_editor"
    ) -> Dict:
        """Main entry point for content editing"""
        
        # Analyze current state
        content, rhetoric = self.analyzer.analyze(text)
        state = np.concatenate([
            content.to_array(),
            rhetoric.to_array(),
            np.array([0.5, 0.5, 0.3, 0.6])  # platform context
        ])
        
        # Decide next action (simplified - would use trained policy)
        action = self._select_action(state)
        
        # Generate multiple candidates via LLM
        candidate_texts = await self.llm_generator.generate_candidates(
            text, action, n_candidates=3
        )
        
        # Analyze each candidate
        candidates_with_state = []
        for candidate_text in candidate_texts:
            cand_content, cand_rhetoric = self.analyzer.analyze(candidate_text)
            cand_state = np.concatenate([
                cand_content.to_array(),
                cand_rhetoric.to_array(),
                np.array([0.5, 0.5, 0.3, 0.6])
            ])
            
            confidence = self.value_network.score_candidate(state, cand_state, action)
            candidates_with_state.append((candidate_text, cand_state, action, confidence))
        
        # PHASE-SPECIFIC BEHAVIOR
        
        if self.phase == "shadow":
            # Shadow mode: just log, don't show to user
            return {
                'mode': 'shadow',
                'suggestions': candidates_with_state,
                'shown_to_user': False,
                'message': 'Silently learning from human edits'
            }
        
        elif self.phase == "assisted":
            # Assisted mode: show suggestions, human decides
            proposals = self.hitl.suggest_edits(text, state, candidates_with_state, top_k=3)
            
            return {
                'mode': 'assisted',
                'proposals': [p.to_dict() for p in proposals],
                'shown_to_user': True,
                'message': f'Showing {len(proposals)} suggestions'
            }
        
        elif self.phase == "autopilot":
            # Autopilot: use best candidate (with human review for high-stakes)
            best_idx, confidence = self.value_network.select_best_candidate(
                state, 
                [(t, s) for t, s, _, _ in candidates_with_state],
                action
            )
            
            needs_review = confidence < 0.7
            
            return {
                'mode': 'autopilot',
                'selected_edit': candidate_texts[best_idx],
                'confidence': confidence,
                'needs_human_review': needs_review,
                'message': 'Auto-applied' if not needs_review else 'Flagged for review'
            }
    
    def record_human_decision(
        self,
        proposal_id: str,
        accepted: bool,
        human_edit: Optional[str] = None,
        editor_id: str = "demo_editor"
    ):
        """Record human editor's decision"""
        
        feedback = HumanFeedback(
            proposal_id=proposal_id,
            accepted=accepted,
            human_edit=human_edit,
            comments="",
            editor_id=editor_id,
            timestamp=datetime.now().isoformat()
        )
        
        self.hitl.record_feedback(feedback)
        
        # Add to imitation learning dataset
        proposal = next(
            (p for p in self.hitl.proposals_log if p.proposal_id == proposal_id),
            None
        )
        
        if proposal:
            self.imitation_learner.add_demonstration(
                proposal.state_before,
                proposal.action,
                proposal.state_after,
                accepted
            )
    
    def get_metrics(self) -> Dict:
        """Get current system metrics"""
        return {
            'phase': self.phase,
            'agreement_rate': self.hitl.agreement_rate,
            'total_edits_logged': len(self.hitl.proposals_log),
            'training_samples': len(self.imitation_learner.demonstrations),
            'ready_for_rl': len(self.imitation_learner.demonstrations) >= 100
        }
    
    def _select_action(self, state: np.ndarray) -> EditorialAction:
        """Select next editorial action (simplified)"""
        # In production: use trained policy
        # For demo: rule-based selection
        
        if state[2] > 0.7:  # high abstractness
            return EditorialAction.ADD_CONCRETE_CASE
        elif state[0] < 0.4:  # low claim density
            return EditorialAction.SHARPEN_CLAIM
        elif state[5] < 0.4:  # low belief conflict
            return EditorialAction.ADD_COUNTER_ARGUMENT
        else:
            return EditorialAction.CUT_OPENING


class ContentAnalyzer:
    """Analyzes text to extract state features"""
    
    def analyze(self, text: str) -> Tuple[ContentSemantics, RhetoricalQuality]:
        words = text.split()
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        
        claim_indicators = ['should', 'must', 'will', 'is', 'are', 'proves']
        evidence_indicators = ['study', 'data', 'research', 'found', 'showed']
        
        claim_count = sum(1 for word in words if word.lower() in claim_indicators)
        evidence_count = sum(1 for word in words if word.lower() in evidence_indicators)
        
        content = ContentSemantics(
            claim_density=min(1.0, claim_count / max(1, len(words) / 100)),
            evidence_to_claim_ratio=evidence_count / max(1, claim_count),
            abstractness_score=0.5,
            redundancy_score=0.3,
            concrete_example_count=text.count('example') + text.count('instance')
        )
        
        rhetoric = RhetoricalQuality(
            belief_conflict_strength=0.5,
            narrative_tension=0.4,
            hook_strength=min(1.0, len(words[:20]) / 20),
            logical_coherence=0.6,
            cognitive_load=min(1.0, len(words) / sentences / 20)
        )
        
        return content, rhetoric


# ============================================================================
# 8. DEMO: PHASE 1 - SHADOW MODE
# ============================================================================

async def demo_shadow_mode():
    """
    Phase 1: Shadow Mode (2-3 months)
    - Agent suggests edits silently
    - Humans edit normally
    - Track agreement rate
    """
    
    print("="*60)
    print("PHASE 1: SHADOW MODE")
    print("="*60)
    print("Agent learns by observing human edits without interfering\n")
    
    system = ProductionEditorialSystem(phase="shadow")
    
    # Simulate 5 human editing sessions
    test_cases = [
        ("AI might be useful for businesses.", "AI transforms business operations through automation."),
        ("Some people think climate change is important.", "Climate scientists agree: global temperatures are rising at unprecedented rates."),
        ("The study could suggest interesting results.", "The study demonstrates a 34% improvement in patient outcomes."),
        ("There are various approaches to this problem.", "Three proven strategies address this challenge: automation, training, and redesign."),
        ("This might be worth considering.", "Companies adopting this approach see 2x ROI within 18 months.")
    ]
    
    for i, (original, human_edited) in enumerate(test_cases, 1):
        print(f"\nSession {i}")
        print(f"Original: {original}")
        print(f"Human edited to: {human_edited}")
        
        result = await system.process_content(original, editor_id=f"editor_{i}")
        
        # Simulate: compare agent suggestion to human edit
        if result['suggestions']:
            agent_edit = result['suggestions'][0][0]  # best candidate
            print(f"Agent suggested: {agent_edit[:100]}...")
            
            # Log in shadow mode
            system.hitl.shadow_mode(
                original,
                agent_edit,
                EditorialAction.SHARPEN_CLAIM,
                human_edited,
                f"editor_{i}"
            )
    
    # Show metrics
    metrics = system.get_metrics()
    print(f"\n{'='*60}")
    print("SHADOW MODE METRICS")
    print(f"{'='*60}")
    print(f"Total sessions: {metrics['total_edits_logged']}")
    print(f"Agreement rate: {metrics['agreement_rate']:.1%}")
    print(f"Training samples collected: {metrics['training_samples']}")
    print(f"Ready for next phase: {metrics['ready_for_rl']}")


async def demo_assisted_mode():
    """
    Phase 2: Assisted Mode
    - Show suggestions to editors
    - Track acceptance rate
    """
    
    print("\n" + "="*60)
    print("PHASE 2: ASSISTED MODE")
    print("="*60)
    print("Agent shows suggestions, human decides\n")
    
    system = ProductionEditorialSystem(phase="assisted")
    
    text = "AI might be important. Some people think it could change things."
    
    result = await system.process_content(text, editor_id="editor_demo")
    
    print(f"Original text: {text}\n")
    print(f"Agent proposals ({len(result['proposals'])}):\n")
    
    for i, proposal in enumerate(result['proposals'], 1):
        print(f"{i}. {proposal['action']}")
        print(f"   Confidence: {proposal['confidence']:.2f}")
        print(f"   Edit: {proposal['edited_text'][:100]}...")
        print(f"   Rationale: {proposal['rationale']}\n")
    
    # Simulate human accepting proposal #2
    accepted_proposal = result['proposals'][1]
    system.record_human_decision(
        accepted_proposal['proposal_id'],
        accepted=True,
        editor_id="editor_demo"
    )
    
    print(f"Human accepted proposal #{2}")
    print(f"System learning from this preference...")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("END-TO-END RL EDITORIAL SYSTEM")
    print("Option B: LLM Candidates + RL Selection + HITL")
    print("="*60)
    
    # Run Phase 1: Shadow Mode
    asyncio.run(demo_shadow_mode())
    
    # Run Phase 2: Assisted Mode
    asyncio.run(demo_assisted_mode())
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Deploy shadow mode for 2-3 months
   - Log 1000+ human editing sessions
   - Track agreement rate (target: >60%)
   
2. Train imitation policy
   - Bootstrap from human demonstrations
   - Validate on held-out edits
   
3. Switch to assisted mode
   - Show top-3 suggestions to editors
   - Track acceptance rate (target: >70%)
   - Collect preference data for RL
   
4. Fine-tune with RL
   - Use PPO with human feedback
   - Optimize for long-term engagement metrics
   
5. Gradual autopilot rollout
   - Start with low-stakes content (social media)
   - Require human review for published articles
   - Monitor quality metrics continuously
    """)