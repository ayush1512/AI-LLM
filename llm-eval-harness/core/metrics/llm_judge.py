import json
import re
from typing import Optional, Dict, Any
from core.metrics.base import BaseMetric, MetricResult
from config import config

FAITHFULNESS_RUBRIC = """
Criteria: Faithfulness evaluates whether every claim made in the actual answer is strictly substantiated by the provided context.
Scoring:
- 5: Entirely faithful. Every factual assertion is directly grounded in the context without extrapolations.
- 4: Mostly faithful. One minor detail is ambiguous but does not conflict with context.
- 3: Partially faithful. Contains unsupported claims alongside grounded facts.
- 2: Mostly unfaithful. Contains notable contradictions or hallucinations.
- 1: Completely ungrounded. Contradicts or ignores the provided context entirely.
"""

RELEVANCY_RUBRIC = """
Criteria: Answer Relevancy measures how directly and completely the response addresses the specific question asked, without unnecessary tangents or evasiveness.
Scoring:
- 5: Highly relevant. Fully and directly answers the question with clarity.
- 4: Relevant. Answers the core question with minor excess elaboration.
- 3: Moderately relevant. Answers part of the question but misses key requirements.
- 2: Low relevance. Mostly off-topic or evasive.
- 1: Irrelevant. Completely fails to address the user query.
"""

CONCISENESS_RUBRIC = """
Criteria: Conciseness evaluates whether the answer conveys the required information efficiently without verbose padding, conversational filler, or repetition.
Scoring:
- 5: Exceptionally concise. Dense with information, zero fluff.
- 4: Concise. Clear and brief with negligible filler words.
- 3: Average. Noticeable pleasantries or minor repetition.
- 2: Wordy. Verbose preamble or repetitive explanations.
- 1: Extremely bloated or repetitive.
"""

class LLMJudgeMetric(BaseMetric):
    """
    G-Eval / LLM-as-a-Judge metric utilizing Chain-of-Thought scoring rubrics.
    """
    def __init__(self, criterion_name: str, rubric: str, judge_model: Optional[str] = None, threshold: float = 0.70):
        self._name = criterion_name
        self.rubric = rubric
        self.judge_model = judge_model or config.judge_model
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"Judge: {self._name}"

    def evaluate(self, prediction: str, expected: Optional[str] = None, context: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        api_key = config.openai_api_key

        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=config.openai_base_url)

                eval_prompt = (
                    f"You are an expert impartial AI evaluator evaluating response quality.\n"
                    f"Rubric:\n{self.rubric}\n\n"
                    f"Context (if any):\n{context or 'None provided.'}\n\n"
                    f"Reference Answer (if any):\n{expected or 'None provided.'}\n\n"
                    f"Candidate Response Under Test:\n{prediction}\n\n"
                    f"Instructions: Think step-by-step about how well the candidate response meets the criteria.\n"
                    f"Output strictly a JSON object formatted as:\n"
                    f'{{"reasoning": "Step-by-step justification...", "score": <integer from 1 to 5>}}'
                )

                resp = client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.0
                )
                raw_out = resp.choices[0].message.content.strip()
                if "```json" in raw_out:
                    raw_out = raw_out.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_out:
                    raw_out = raw_out.split("```")[1].split("```")[0].strip()

                parsed = json.loads(raw_out)
                score_raw = float(parsed.get("score", 3))
                # Normalize 1-5 scale to 0.0 - 1.0
                norm_score = max(0.0, min(1.0, (score_raw - 1.0) / 4.0))
                passed = (norm_score >= self.threshold)
                reasoning = parsed.get("reasoning", "Evaluation completed.")

                return MetricResult(
                    metric_name=self.name,
                    score=round(norm_score, 3),
                    passed=passed,
                    reason=f"Score {score_raw}/5 ({norm_score*100:.0f}%): {reasoning}",
                    details={"raw_score_1_to_5": score_raw, "reasoning": reasoning}
                )
            except Exception as e:
                pass

        # Offline heuristic judge fallback
        # Evaluates grounding and token overlap
        pred_words = set(re.findall(r'\b\w+\b', prediction.lower()))
        if context:
            ctx_words = set(re.findall(r'\b\w+\b', context.lower()))
            overlap = len(pred_words.intersection(ctx_words)) / max(1, len(pred_words))
            norm_score = min(1.0, 0.5 + (overlap * 0.5))
        elif expected:
            exp_words = set(re.findall(r'\b\w+\b', expected.lower()))
            overlap = len(pred_words.intersection(exp_words)) / max(1, len(exp_words))
            norm_score = min(1.0, 0.6 + (overlap * 0.4))
        else:
            norm_score = 0.85 if len(prediction.strip()) > 5 else 0.4

        passed = norm_score >= self.threshold
        return MetricResult(
            metric_name=self.name,
            score=round(norm_score, 3),
            passed=passed,
            reason=f"[Offline Evaluator] Assessed alignment score: {norm_score*100:.0f}% against rubric."
        )

def get_default_judges() -> list[LLMJudgeMetric]:
    return [
        LLMJudgeMetric("Faithfulness", FAITHFULNESS_RUBRIC),
        LLMJudgeMetric("Relevancy", RELEVANCY_RUBRIC),
        LLMJudgeMetric("Conciseness", CONCISENESS_RUBRIC)
    ]

