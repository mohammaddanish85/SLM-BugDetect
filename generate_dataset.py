"""Deterministically generate the SLM-BugDetect synthetic dataset (Sec. 3.2).

Outputs:
    data/dataset.jsonl  -- 100 instruction-response pairs (training)
    data/heldout.jsonl  -- 6 held-out pairs (evaluation, Table 1)
"""
import json, os, random

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = "Analyze this Python code for bugs: {code}"

NAMES = ["a", "b", "x", "y", "total", "count", "value", "num", "den", "score"]
FUNCS = ["divide", "ratio", "avg", "scale", "normalize", "split_bill", "quotient"]
LISTS = ["nums", "items", "values", "scores", "data"]


def make_zero_div(rng):
    f = rng.choice(FUNCS); a, b = rng.sample(NAMES, 2)
    code = f"def {f}({a}, {b}):\n    return {a} / 0"
    fix = (f"def {f}({a}, {b}):\n    if {b} == 0:\n"
           f"        raise ZeroDivisionError('division by zero')\n"
           f"    return {a} / {b}")
    return code, fix, "ZeroDivisionError", "raise ZeroDivisionError"


def make_index(rng):
    lst = rng.choice(LISTS); n = rng.randint(2, 5); k = n + rng.randint(1, 3)
    elems = ", ".join(str(rng.randint(1, 9)) for _ in range(n))
    code = f"{lst} = [{elems}]\nprint({lst}[{k}])"
    fix = (f"{lst} = [{elems}]\nif {k} < len({lst}):\n    print({lst}[{k}])\n"
           f"else:\n    print('index out of range')")
    return code, fix, "IndexError", f"len({lst})"


def make_syntax(rng):
    if rng.random() < 0.6:                      # missing colon
        v = rng.choice(["i", "j", "n", "idx"]); m = rng.randint(3, 12)
        code = f"for {v} in range({m})\n    print({v})"
        fix = f"for {v} in range({m}):\n    print({v})"
        return code, fix, "SyntaxError", f"range({m}):"
    v = rng.choice(NAMES); k = rng.randint(2, 9)   # = instead of ==
    code = f"if {v} = {k}:\n    print({v})"
    fix = f"if {v} == {k}:\n    print({v})"
    return code, fix, "SyntaxError", f"{v} == {k}:"


def build(n, makers, rng):
    out, seen = [], set()
    while len(out) < n:
        code, fix, fault, chk = rng.choice(makers)(rng)
        if code in seen:
            continue
        seen.add(code)
        out.append({"fault": fault, "code": code, "fix": fix, "must_contain": chk,
                    "prompt": PROMPT.format(code=code),
                    "target": f"The code contains a {fault}. Corrected version:\n{fix}"})
    return out


def main():
    rng = random.Random(SEED)
    rows = (build(34, [make_zero_div], rng)
            + build(33, [make_index], rng)
            + build(33, [make_syntax], rng))
    for i, r in enumerate(rows):
        r["id"] = i
    with open(os.path.join(HERE, "dataset.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    heldout = [
        {"id": 0, "fault": "ZeroDivisionError", "must_contain": "raise ZeroDivisionError",
         "code": "def divide(a, b):\n    return a / 0",
         "fix": "def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('division by zero')\n    return a / b"},
        {"id": 1, "fault": "SyntaxError", "must_contain": "range(10):",
         "code": "for i in range(10)\n    print(i)",
         "fix": "for i in range(10):\n    print(i)"},
        {"id": 2, "fault": "KeyError", "must_contain": ".get(",
         "code": "my_dict = {'a': 1}\nprint(my_dict['b'])",
         "fix": "print(my_dict.get('b'))"},
        {"id": 3, "fault": "IndexError", "must_contain": "len(nums)",
         "code": "nums = [1, 2]\nprint(nums[5])",
         "fix": "if 5 < len(nums):\n    print(nums[5])"},
        {"id": 4, "fault": "ZeroDivisionError", "must_contain": "raise ZeroDivisionError",
         "code": "def ratio(x, y):\n    return x / 0",
         "fix": "def ratio(x, y):\n    if y == 0:\n        raise ZeroDivisionError('division by zero')\n    return x / y"},
        {"id": 5, "fault": "SyntaxError", "must_contain": "range(7):",
         "code": "total = 0\nfor j in range(7)\n    total += j",
         "fix": "for j in range(7):\n    total += j"},
    ]
    for r in heldout:
        r["prompt"] = PROMPT.format(code=r["code"])
        r["target"] = f"The code contains a {r['fault']}. Corrected version:\n{r['fix']}"
    with open(os.path.join(HERE, "heldout.jsonl"), "w", encoding="utf-8") as fh:
        for r in heldout:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} train + {len(heldout)} held-out samples.")


if __name__ == "__main__":
    main()