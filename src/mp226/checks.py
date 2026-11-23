"""The seven data checks.

Each function looks at a table and answers ONE question from
findings.yaml, returning a dictionary with the same keys the form uses.
Fill in the bodies yourself: the numbers must come from your own code
running on your own file.

Run all of them with:   python -m mp226.checks
"""

from .data import load_raw


def check_leaking_column(train, holdout, target):
    """A column that is only known AFTER the event.

    Return {"present": bool, "column": str or None}
    """
    raise NotImplementedError


def check_unit_mixup(train, later, numeric_cols, group_col):
    """One group value recorded one column in another unit.

    Return {"present": bool, "column": ..., "group_value": ..., "factor": ...}
    """
    raise NotImplementedError


def check_sentinel_value(train, numeric_cols):
    """A special number typed instead of a blank reading.

    Return {"present": bool, "column": ..., "value": ..., "count": ...}
    """
    raise NotImplementedError


def check_duplicate_rows(train):
    """The same record appearing more than once.

    Return {"present": bool, "count": number of EXTRA copies}
    """
    raise NotImplementedError


def check_stuck_sensor(train, numeric_cols):
    """A column showing the same value for a long run of days.

    Return {"present": bool, "column": ..., "start_date": ..., "end_date": ...}
    """
    raise NotImplementedError


def check_target_typos(train, target):
    """Targets typed 10 times too big.

    Return {"present": bool, "count": number of such rows}
    """
    raise NotImplementedError


def check_spelling_variants(train, later, text_cols):
    """The same category written in different ways.

    Return {"present": bool, "column": ..., "true_levels": ...}
    """
    raise NotImplementedError


def main():
    from .config import load_params

    cols = load_params()["columns"]
    train, later, holdout = (load_raw(n) for n in ("train", "later", "holdout"))

    # Order matters: tidy the spellings first, or a per-tower check
    # would split one tower into several groups.
    jobs = [
        ("spelling_variants", lambda: check_spelling_variants(train, later, cols["text"])),
        ("leaking_column", lambda: check_leaking_column(train, holdout, cols["target"])),
        ("unit_mixup", lambda: check_unit_mixup(train, later, cols["numeric"], cols["group"])),
        ("sentinel_value", lambda: check_sentinel_value(train, cols["numeric"])),
        ("duplicate_rows", lambda: check_duplicate_rows(train)),
        ("stuck_sensor", lambda: check_stuck_sensor(train, cols["numeric"])),
        ("target_typos", lambda: check_target_typos(train, cols["target"])),
    ]
    for name, job in jobs:
        try:
            print(f"{name:18s}", job())
        except NotImplementedError:
            print(f"{name:18s} TODO")


if __name__ == "__main__":
    main()
