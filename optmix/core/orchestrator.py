"""
Orchestrator Router — intelligent task routing for the OptMix team.

Routes user messages to the appropriate specialist agent using either
LLM-based intent classification or keyword-based fallback rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from optmix.agents.schema import AgentDefinition

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of routing a user message to an agent."""

    agent_name: str
    confidence: str  # "high", "medium", "low"
    reasoning: str
    method: str  # "llm", "keyword", "default"


# Keyword rules extracted from orchestrator.agent.yaml
KEYWORD_RULES: list[dict[str, str]] = [
    {
        "pattern": "business context|objective|KPI|strategy|competitive|market|brief|measurement plan",
        "agent": "strategist",
        "reason": "Business context and strategy questions go to Maya",
    },
    {
        "pattern": "data quality|EDA|feature|anomaly|collinearity|validate|load data|csv|missing|null",
        "agent": "analyst",
        "reason": "Data-related tasks go to Kai",
    },
    {
        "pattern": "model|fit|bayesian|adstock|saturation|prior|contribution|MMM|regression|coefficient",
        "agent": "modeler",
        "reason": "Modeling tasks go to Priya",
    },
    {
        "pattern": "optimize|budget|allocation|scenario|what if|what-if|marginal|ROAS|reallocate|spend",
        "agent": "optimizer",
        "reason": "Optimization and scenarios go to Ravi",
    },
    {
        "pattern": "report|dashboard|chart|presentation|summary|insight|action plan|executive|visuali",
        "agent": "reporter",
        "reason": "Reporting and visualization go to Nora",
    },
]

# LLM classification prompt for the orchestrator
CLASSIFICATION_PROMPT = """You are the OptMix Orchestrator. Your job is to route user messages to the right specialist agent.

## Available Agents

1. **strategist** (Maya Chen) — Business context, KPIs, measurement strategy, competitive analysis, market context. Route here for strategic questions, objective setting, and "why should we measure this?" questions.

2. **analyst** (Kai Nakamura) — Data loading, validation, EDA, feature engineering, data quality. Route here for data-related tasks, loading CSVs, checking data quality, and exploratory analysis.

3. **modeler** (Priya Sharma) — MMM model fitting, diagnostics, channel contributions, saturation curves, adstock. Route here for anything about building, fitting, or interpreting marketing mix models.

4. **optimizer** (Ravi Santos) — Budget allocation, scenario simulation, marginal ROAS, constrained optimization. Route here for "how should we allocate budget?" and "what if we change X?" questions.

5. **reporter** (Nora Lindqvist) — Reports, dashboards, charts, executive summaries, action plans. Route here for generating deliverables and visualizations.

## Rules
- If the message is broad (e.g., "help me understand my marketing data"), start with the **strategist**
- If the message mentions data/CSV/loading, route to **analyst**
- If the message is about a complete analysis, route to **strategist** (they'll coordinate)
- When in doubt, prefer **strategist** — Maya asks the right clarifying questions

## Respond with ONLY a JSON object:
{"agent": "agent_name", "confidence": "high|medium|low", "reasoning": "one sentence why"}"""


class OrchestratorRouter:
    """
    Routes user messages to the appropriate specialist agent.

    Uses LLM-based intent classification with keyword fallback.
    """

    def __init__(
        self,
        agents: dict[str, AgentDefinition],
        llm_client: Any | None = None,
    ) -> None:
        self.agents = agents
        self.llm_client = llm_client

        # Load routing rules from orchestrator agent if available
        orchestrator = agents.get("orchestrator")
        if orchestrator and orchestrator.routing_rules:
            self._keyword_rules = [
                {
                    "pattern": rule.get("pattern", ""),
                    "agent": rule["agent"],
                    "reason": rule.get("reason", ""),
                }
                for rule in orchestrator.routing_rules
            ]
        else:
            self._keyword_rules = KEYWORD_RULES

    def route(self, message: str, state: Any = None) -> RoutingDecision:
        """
        Route a user message to the best agent.

        Tries LLM classification first, falls back to keyword matching.

        Args:
            message: The user's message.
            state: Current shared state (can influence routing).

        Returns:
            RoutingDecision with the target agent and reasoning.
        """
        # Try LLM-based routing first
        if self.llm_client is not None:
            try:
                decision = self._route_with_llm(message, state)
                if decision and decision.agent_name in self.agents:
                    return decision
            except Exception as e:
                logger.warning("LLM routing failed, falling back to keywords: %s", e)

        # Fallback to keyword matching
        decision = self._route_with_keywords(message)
        if decision:
            return decision

        # Default fallback
        return RoutingDecision(
            agent_name="strategist",
            confidence="low",
            reasoning="No strong signal detected. Routing to Strategist as the default entry point.",
            method="default",
        )

    def _route_with_llm(self, message: str, state: Any = None) -> RoutingDecision | None:
        """Use the LLM to classify intent and select an agent."""
        import json

        state_context = ""
        if state and hasattr(state, "to_summary"):
            state_context = f"\n\n## Current State\n{state.to_summary()}"

        system = CLASSIFICATION_PROMPT + state_context

        response = self.llm_client.chat(
            system=system,
            messages=[{"role": "user", "content": message}],
            max_tokens=256,
            temperature=0.0,
        )

        # Parse JSON response
        try:
            text = response.content.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            parsed = json.loads(text)
            agent_name = parsed.get("agent", "strategist")
            confidence = parsed.get("confidence", "medium")
            reasoning = parsed.get("reasoning", "LLM classification")

            return RoutingDecision(
                agent_name=agent_name,
                confidence=confidence,
                reasoning=reasoning,
                method="llm",
            )
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.warning("Failed to parse LLM routing response: %s", e)
            return None

    def _route_with_keywords(self, message: str) -> RoutingDecision | None:
        """Route using keyword pattern matching."""
        message_lower = message.lower()

        best_match: dict[str, Any] | None = None
        best_score = 0

        for rule in self._keyword_rules:
            patterns = rule["pattern"].split("|")
            score = sum(1 for p in patterns if p.strip().lower() in message_lower)
            if score > best_score:
                best_score = score
                best_match = rule

        if best_match and best_score > 0:
            return RoutingDecision(
                agent_name=best_match["agent"],
                confidence="high" if best_score >= 2 else "medium",
                reasoning=best_match.get("reason", f"Matched {best_score} keyword(s)"),
                method="keyword",
            )

        return None
