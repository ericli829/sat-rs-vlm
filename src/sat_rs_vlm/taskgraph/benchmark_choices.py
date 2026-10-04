"""Legacy benchmark option identifiers, assigned before generic resolution."""


def benchmark_choice_ids(options: tuple[str, ...]) -> tuple[str, ...]:
    if len(options) > 26:
        raise ValueError("benchmark choices support at most 26 options")
    return tuple(chr(ord("A") + index) for index in range(len(options)))
