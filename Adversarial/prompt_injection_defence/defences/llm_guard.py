from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType

scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)


def llm_guard_prompt_injection_detection(_, prompt):

    sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
    if is_valid:
        return 0
    else:
        return 1