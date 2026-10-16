"""Measure transcription fidelity without treating reformatted math as exact."""


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def score(predictions, references):
    predictions, references = list(predictions), list(references)
    if not references or len(predictions) != len(references):
        raise ValueError("Need equally sized, nonempty predictions and references")
    if any(not isinstance(value, str) for value in predictions + references):
        raise ValueError("Predictions and references must be strings")
    chars = sum(len(value) for value in references)
    if chars == 0:
        raise ValueError("Reference text must contain characters")
    return {
        "examples": len(references),
        "exact_match": sum(p == r for p, r in zip(predictions, references)) / len(references),
        "whitespace_ignored_match": sum("".join(p.split()) == "".join(r.split())
                                         for p, r in zip(predictions, references)) / len(references),
        "character_error_rate": sum(edit_distance(p, r)
                                     for p, r in zip(predictions, references)) / chars,
    }
