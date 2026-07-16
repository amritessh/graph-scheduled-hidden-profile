"""System prompts for reasoning task agents."""

from __future__ import annotations

from gshp.task.reasoning import ReasoningProblem


def agent_system_prompt_reasoning(agent_id: int, problem: ReasoningProblem) -> str:
    return (
        f"You are Agent {agent_id}, a member of a quantitative reasoning panel. "
        f"Your panel is working together to solve the following problem:\n\n"
        f"--- PROBLEM ---\n{problem.problem_text}\n--- END PROBLEM ---\n\n"
        "Discuss with your colleagues to reason through it carefully and reach the correct answer. "
        "You all have exactly the same information — the problem above. "
        "Do not invent additional facts. Keep responses concise."
    )


def final_user_prompt_reasoning(agent_id: int, memory: str, problem: ReasoningProblem) -> str:
    return (
        f"You are Agent {agent_id}. Here is a log of your panel discussions:\n\n"
        f"{memory if memory.strip() else '(You had no conversations yet.)'}\n\n"
        f"The problem your panel was solving:\n{problem.problem_text}\n\n"
        "Based on your reasoning and discussions, provide your final answer.\n"
        "Reply with ONLY a single JSON object, no markdown fences:\n"
        '{"answer":"YOUR_ANSWER","justification":"one short sentence"}\n'
        "Provide the most specific answer possible (an integer, fraction, or decimal). "
        "Do not include units or currency symbols in 'answer'."
    )


def deliberation_user_prompt_reasoning(
    agent_id: int,
    memory: str,
    individual_decisions: list,
    problem: ReasoningProblem,
) -> str:
    decisions_block = "\n".join(
        f"  Agent {d.agent_id}: {d.choice or '?'} — {d.justification[:120]}"
        for d in individual_decisions
    )
    mem_section = memory.strip() if memory.strip() else "(No conversations recorded.)"
    return (
        f"You are Agent {agent_id}. Here is a log of your conversations:\n\n"
        f"{mem_section}\n\n"
        f"The problem: {problem.problem_text}\n\n"
        "After all discussions, each panel member submitted their answer:\n"
        f"{decisions_block}\n\n"
        "Considering all reasoning shared, what is your final answer?\n"
        "Reply with ONLY a single JSON object, no markdown fences:\n"
        '{"answer":"YOUR_ANSWER","justification":"one short sentence"}\n'
        "Do not include units or currency symbols in 'answer'."
    )
