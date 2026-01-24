import asyncio
import random
import math
import re
from enum import IntEnum
from typing import List, Dict, Any

import aiohttp
from openai import OpenAI

# =============================================================================
# STATE REPRESENTATION
# =============================================================================

class ContentState:
    def __init__(self):
        self.researchCompleteness = 0.0
        self.outlineCompleteness = 0.0
        self.draftCompleteness = 0.0
        self.hookStrength = 0.0
        self.clarityScore = 0.0
        self.evidenceDensity = 0.0
        self.narrativeFlow = 0.0
        self.platformFit = 0.0
        self.currentStage = 0
        self.revisionCount = 0
        self.tokensUsed = 0

        self.researchData: List[str] = []
        self.outline: List[str] = []
        self.hook: str = ""
        self.sections: List[str] = []
        self.fullDraft: str = ""

    def to_vector(self) -> List[float]:
        return [
            self.researchCompleteness,
            self.outlineCompleteness,
            self.draftCompleteness,
            self.hookStrength,
            self.clarityScore,
            self.evidenceDensity,
            self.narrativeFlow,
            self.platformFit,
            self.currentStage / 4.0,
            min(self.revisionCount / 10.0, 1.0),
            min(self.tokensUsed / 10000.0, 1.0),
        ]

    def quality_score(self) -> float:
        return (
            self.hookStrength
            + self.clarityScore
            + self.evidenceDensity
            + self.narrativeFlow
            + self.platformFit
        ) / 5.0


# =============================================================================
# ACTION SPACE
# =============================================================================

class ActionType(IntEnum):
    RESEARCH = 0
    CREATE_OUTLINE = 1
    WRITE_HOOK = 2
    WRITE_SECTION = 3
    REFINE_CONTENT = 4
    FINALIZE = 5


# =============================================================================
# OPENAI-COMPATIBLE LLM CALL
# =============================================================================


client = OpenAI(
    api_key="sk-rmoLqMBRKQ08WHiKikzRjEPEZipyCAviNkkNLTGGauhomeZX",          # replace with your key
    base_url="https://api.moonshot.cn/v1",
)

def call_llm(prompt: str, max_tokens: int = 8000) -> str:
    completion = client.chat.completions.create(
        model="kimi-k2-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 Kimi，由 Moonshot AI 提供的人工智能助手。"
                    "你擅长中文和英文内容创作，逻辑清晰，表达专业。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.6,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content.strip()



# =============================================================================
# ENVIRONMENT
# =============================================================================

class ContentCreationEnvironment:
    def __init__(self, topic: str, platform="wechat", max_steps=15):
        self.topic = topic
        self.platform = platform
        self.max_steps = max_steps
        self.current_step = 0
        self.state = ContentState()
        self.target_sections = 5

    def reset(self) -> ContentState:
        self.current_step = 0
        self.state = ContentState()
        return self.state

    async def execute_research(self, intensity: float):
        prompt = f"""
Research topic: "{self.topic}"

Provide {math.ceil(intensity * 5)} key findings.
Use bullets. Include data or statistics.
"""
        text = call_llm(prompt)
        bullets = [l for l in text.split("\n") if l.strip().startswith(("-", "•", "*"))]
        self.state.researchData.extend(bullets)

        has_numbers = bool(re.search(r"\d+%|\$\d+", text))
        improvement = intensity * 0.3

        self.state.researchCompleteness = min(1.0, self.state.researchCompleteness + improvement)
        self.state.evidenceDensity = min(
            1.0, self.state.evidenceDensity + (0.3 if has_numbers else 0.1)
        )

        self.state.tokensUsed += len(text.split()) * 1.3
        return improvement

    async def execute_outline(self, intensity: float):
        context = "\n".join(self.state.researchData[:10])
        prompt = f"""
Create outline for "{self.topic}"

Research:
{context}

{self.target_sections} sections, numbered.
"""
        outline = await call_llm(prompt)
        sections = [s.strip() for s in re.split(r"\n(?=\d+\.)", outline) if s.strip()]
        self.state.outline = sections

        improvement = intensity * 0.4
        self.state.outlineCompleteness = min(1.0, self.state.outlineCompleteness + improvement)
        self.state.narrativeFlow = min(1.0, self.state.narrativeFlow + improvement * 0.6)
        self.state.tokensUsed += 500
        self.state.currentStage = max(self.state.currentStage, 1)
        return improvement

    async def execute_hook(self, intensity: float):
        context = "\n".join(self.state.researchData[:5])
        prompt = f"""
Write a hook for "{self.topic}"

Context:
{context}

2–3 sentences. Start strong.
"""
        hook = await call_llm(prompt)
        self.state.hook = hook

        quality = (
            (0.3 if "?" in hook else 0)
            + (0.4 if re.search(r"\d+%|\$\d+", hook) else 0)
            + (0.3 if len(hook.split()) < 50 else 0.1)
        )

        self.state.hookStrength = min(
            1.0, self.state.hookStrength + intensity * 0.5 + quality * 0.3
        )
        self.state.draftCompleteness += 0.15
        self.state.tokensUsed += 200
        self.state.currentStage = max(self.state.currentStage, 2)

        return quality

    async def execute_section(self, idx: int, intensity: float):
        outline = self.state.outline[idx] if idx < len(self.state.outline) else f"Section {idx+1}"
        context = "\n".join(self.state.researchData[:8])

        prompt = f"""
Write section for "{self.topic}"

{outline}

Context:
{context}

150–200 words.
"""
        text = await call_llm(prompt)

        while len(self.state.sections) <= idx:
            self.state.sections.append("")
        self.state.sections[idx] = text

        wc = len(text.split())
        quality = (
            (0.3 if wc >= 100 else 0.1)
            + (0.3 if re.search(r"example|case|such as", text, re.I) else 0)
            + (0.4 if re.search(r"\d+%|\$\d+", text) else 0)
        )

        self.state.draftCompleteness = len([s for s in self.state.sections if s]) / self.target_sections
        self.state.clarityScore = min(1.0, self.state.clarityScore + quality * 0.2)
        self.state.tokensUsed += wc * 1.3
        return quality

    async def step(self, action: Dict[str, Any]):
        self.current_step += 1
        reward = 0.0

        t = action["actionType"]
        intensity = action.get("intensity", 0.5)
        idx = action.get("targetSection", 0)

        if t == ActionType.RESEARCH:
            reward += 0.1 * await self.execute_research(intensity)

        elif t == ActionType.CREATE_OUTLINE and self.state.researchCompleteness > 0.3:
            reward += 0.15 * await self.execute_outline(intensity)

        elif t == ActionType.WRITE_HOOK and self.state.outlineCompleteness > 0.4:
            reward += 0.2 * await self.execute_hook(intensity)

        elif t == ActionType.WRITE_SECTION and self.state.outlineCompleteness > 0.5:
            reward += 0.15 + await self.execute_section(idx, intensity) * 0.1

        elif t == ActionType.FINALIZE:
            self.state.fullDraft = self.state.hook + "\n\n" + "\n\n".join(self.state.sections)
            quality = self.state.quality_score()
            reward += 1.0 + quality if quality > 0.6 else 0.5
            self.state.currentStage = 4

        if self.state.tokensUsed > 8000:
            reward -= 0.1

        done = self.state.currentStage == 4 or self.current_step >= self.max_steps
        return self.state, reward, done


# =============================================================================
# SIMPLE Q-LEARNING POLICY
# =============================================================================

class SimplePolicy:
    def __init__(self, state_dim: int):
        self.lr = 0.01
        self.gamma = 0.95
        self.epsilon = 0.5
        self.weights = {
            a: [(random.random() - 0.5) * 0.01 for _ in range(state_dim)]
            for a in ActionType
        }

    def q(self, s, action):
        w = self.weights[action["actionType"]]
        return sum(x * wi for x, wi in zip(s, w)) * action.get("intensity", 1.0)

    def act(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        s = state.to_vector()
        return max(actions, key=lambda a: self.q(s, a))

    def update(self, s, a, r, s2, actions2, done):
        sv = s.to_vector()
        q = self.q(sv, a)

        target = r if done else r + self.gamma * max(self.q(s2.to_vector(), a2) for a2 in actions2)
        td = target - q

        w = self.weights[a["actionType"]]
        for i in range(len(w)):
            w[i] += self.lr * td * sv[i] * a.get("intensity", 1.0)

    def decay(self):
        self.epsilon = max(0.05, self.epsilon * 0.99)


# =============================================================================
# TRAIN / DEMO
# =============================================================================

async def demo():
    env = ContentCreationEnvironment("Why 90% of AI Products Fail After Launch")
    policy = SimplePolicy(len(env.state.to_vector()))

    for _ in range(5):
        s = env.reset()
        done = False
        while not done:
            actions = env.get_valid_actions() if hasattr(env, "get_valid_actions") else [
                {"actionType": ActionType.RESEARCH, "intensity": 0.5},
                {"actionType": ActionType.FINALIZE, "intensity": 1.0},
            ]
            a = policy.act(s, actions)
            s2, r, done = await env.step(a)
            policy.update(s, a, r, s2, actions, done)
            s = s2
        policy.decay()

    print("\n=== FINAL DRAFT ===\n")
    print(env.state.fullDraft)


if __name__ == "__main__":
    asyncio.run(demo())
