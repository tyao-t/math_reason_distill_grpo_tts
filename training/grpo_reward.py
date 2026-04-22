from evaluate_math500_vllm import (
    extract_final_candidate,
    grade_answer,
    render_prompt,
)


def compute_reward(generation_text: str, ground_truth_answer: str) -> float:
    extracted = extract_final_candidate(generation_text, fallback=None)
    if not extracted:
        return 0.0
    return float(grade_answer(extracted, ground_truth_answer))


__all__ = [
    "compute_reward",
    "extract_final_candidate",
    "grade_answer",
    "render_prompt",
]
