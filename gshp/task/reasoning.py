"""
Reasoning task for collective problem-solving experiments.

All agents receive identical problem text — no information asymmetry.
Bottleneck is reasoning quality, not information sharing.
Designed as a contrast to the hidden-profile hiring task.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel


class ReasoningProblem(BaseModel):
    problem_id: str
    problem_text: str
    correct_answer: str
    answer_explanation: str
    answer_hint: str = ""


class ReasoningTaskSpec(BaseModel):
    task_id: str = "reasoning_v1"
    problems: list[ReasoningProblem]

    def get_problem(self, problem_id: str) -> ReasoningProblem:
        for p in self.problems:
            if p.problem_id == problem_id:
                return p
        raise KeyError(f"Unknown problem_id: {problem_id}")

    def is_correct(self, problem_id: str, answer: str) -> bool:
        p = self.get_problem(problem_id)
        return _normalize(answer) == _normalize(p.correct_answer)


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[$%,\s]", "", s)
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            val = float(num) / float(den)
        else:
            val = float(s)
        return f"{val:.2f}"
    except (ValueError, ZeroDivisionError):
        return s


def build_default_reasoning_task() -> ReasoningTaskSpec:
    problems = [
        ReasoningProblem(
            problem_id="bayes_factory",
            problem_text=(
                "A factory produces electronic components. 3% of components are defective. "
                "A quality sensor correctly identifies 92% of defective components as faulty "
                "(sensitivity = 92%), but incorrectly flags 5% of non-defective components as faulty "
                "(false positive rate = 5%). "
                "A component is flagged as faulty by the sensor. "
                "What is the probability, to the nearest whole percent, "
                "that this component is actually defective? "
                "Provide your answer as an integer (e.g. '36')."
            ),
            correct_answer="36",
            answer_explanation=(
                "P(defective|flagged) = (0.92×0.03) / (0.92×0.03 + 0.05×0.97) "
                "= 0.0276 / 0.0761 ≈ 36%"
            ),
            answer_hint="Apply Bayes' theorem with the given base rate.",
        ),
        ReasoningProblem(
            problem_id="conditional_balls",
            problem_text=(
                "A bag contains 4 red balls and 6 blue balls. "
                "Two balls are drawn one at a time without replacement. "
                "Given that the first ball drawn was red, "
                "what is the probability that the second ball is also red? "
                "Express as a simplified fraction (e.g. '1/3')."
            ),
            correct_answer="1/3",
            answer_explanation=(
                "After drawing one red ball, 3 red and 6 blue remain (9 total). "
                "P(2nd red | 1st red) = 3/9 = 1/3."
            ),
        ),
        ReasoningProblem(
            problem_id="compound_interest",
            problem_text=(
                "An investment of $10,000 grows at 8% per year compounded quarterly. "
                "After exactly 5 years, what is the balance to the nearest dollar? "
                "Provide your answer as an integer with no dollar sign (e.g. '14859')."
            ),
            correct_answer="14859",
            answer_explanation=(
                "A = 10000 × (1 + 0.08/4)^(4×5) = 10000 × (1.02)^20 "
                "= 10000 × 1.48594... ≈ $14,859."
            ),
        ),
        ReasoningProblem(
            problem_id="expected_value_dice",
            problem_text=(
                "A game costs $6 to play. You roll two standard six-sided dice. "
                "If the sum is 7, you win $25. "
                "If the sum is 2 or 12, you win $40. "
                "Otherwise you win nothing. "
                "What is the expected profit (positive) or loss (negative) per game, "
                "rounded to the nearest cent? "
                "Provide your answer as a decimal (e.g. '+0.39' or '-1.25')."
            ),
            correct_answer="0.39",
            answer_explanation=(
                "P(sum=7)=6/36, P(sum=2 or 12)=2/36. "
                "EV of winnings = (6/36)×25 + (2/36)×40 = 150/36 + 80/36 = 230/36 ≈ 6.39. "
                "Expected profit = 6.39 - 6.00 = +$0.39."
            ),
        ),
        ReasoningProblem(
            problem_id="tsp_three_cities",
            problem_text=(
                "A delivery driver starts at a depot and must visit exactly three stops "
                "(A, B, C) before returning to the depot. "
                "Travel costs (symmetric): Depot-A=4, Depot-B=2, Depot-C=6, "
                "A-B=3, A-C=5, B-C=7. "
                "What is the minimum total travel cost for the round trip? "
                "Provide your answer as an integer."
            ),
            correct_answer="16",
            answer_explanation=(
                "All 6 routes: "
                "D→A→B→C→D=20, D→A→C→B→D=18, D→B→A→C→D=16, "
                "D→B→C→A→D=18, D→C→A→B→D=16, D→C→B→A→D=20. "
                "Minimum = 16 (routes D→B→A→C→D or D→C→A→B→D)."
            ),
        ),
        ReasoningProblem(
            problem_id="pipeline_scheduling",
            problem_text=(
                "A factory produces items through 3 sequential stages: "
                "Cutting, Assembly, Finishing. Each stage takes exactly 1 hour per item. "
                "Each stage has one machine. As soon as a machine finishes an item, "
                "it immediately starts the next. Items move to the next stage as soon as ready. "
                "How many total hours does it take to fully complete all 10 items "
                "(from when item 1 enters Cutting until item 10 exits Finishing)? "
                "Provide your answer as an integer."
            ),
            correct_answer="12",
            answer_explanation=(
                "Item n finishes Finishing at hour n+2. "
                "Item 10 finishes at hour 10+2 = 12."
            ),
        ),
        ReasoningProblem(
            problem_id="bayes_double_positive",
            problem_text=(
                "A disease affects 0.5% of the population. "
                "A diagnostic test has 98% sensitivity (true positive rate) "
                "and 96% specificity (true negative rate = 96%, so false positive rate = 4%). "
                "A person takes the test on two separate days and tests positive both times. "
                "Assume the two tests are conditionally independent given disease status. "
                "What is the probability, to the nearest whole percent, "
                "that this person actually has the disease? "
                "Provide your answer as an integer."
            ),
            correct_answer="75",
            answer_explanation=(
                "After test 1: P(D|+) = (0.98×0.005)/(0.98×0.005+0.04×0.995) ≈ 10.96%. "
                "Update prior to 10.96%, run test 2: "
                "P(D|++) = (0.98×0.1096)/(0.98×0.1096+0.04×0.8904) ≈ 75%."
            ),
        ),
        ReasoningProblem(
            problem_id="geometric_series",
            problem_text=(
                "The first term of a geometric sequence is 2, "
                "and each subsequent term is 3 times the previous term. "
                "What is the sum of the first 7 terms? "
                "Provide your answer as an integer."
            ),
            correct_answer="2186",
            answer_explanation=(
                "S = 2×(3^7 - 1)/(3 - 1) = 2×(2187 - 1)/2 = 2186."
            ),
        ),
        ReasoningProblem(
            problem_id="work_rate",
            problem_text=(
                "Pipe A fills a tank in 4 hours. "
                "Pipe B fills the same tank in 6 hours. "
                "Pipe C drains the tank in 8 hours. "
                "If all three pipes are open simultaneously starting with an empty tank, "
                "how many hours will it take to fill the tank completely? "
                "Express as a simplified fraction (e.g. '24/7')."
            ),
            correct_answer="24/7",
            answer_explanation=(
                "Combined rate = 1/4 + 1/6 - 1/8 = 6/24 + 4/24 - 3/24 = 7/24 per hour. "
                "Time = 24/7 hours."
            ),
        ),
        ReasoningProblem(
            problem_id="election_bayes",
            problem_text=(
                "In a city, 60% of voters support Candidate A and 40% support Candidate B. "
                "A poll correctly identifies a voter's preference 80% of the time "
                "(for both supporters equally). "
                "A randomly selected voter is polled and says they support Candidate A. "
                "What is the probability, to the nearest whole percent, "
                "that this voter actually supports Candidate A? "
                "Provide your answer as an integer."
            ),
            correct_answer="75",
            answer_explanation=(
                "P(A|says A) = (0.80×0.60) / (0.80×0.60 + 0.20×0.40) "
                "= 0.48 / (0.48 + 0.08) = 0.48/0.56 ≈ 85.7%... "
                "Wait: poll correct 80% means P(says A|A)=0.8, P(says A|B)=0.2. "
                "= 0.48/0.56 ≈ 86%. "
                "Correct answer: 86%."
            ),
            answer_hint="Apply Bayes: P(A|says A) = P(says A|A)×P(A) / P(says A).",
        ),
    ]

    # fix election_bayes correct answer
    for p in problems:
        if p.problem_id == "election_bayes":
            p.correct_answer = "86"
            break

    return ReasoningTaskSpec(problems=problems)
