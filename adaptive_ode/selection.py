"""Selection policies that never use final test targets."""


def cost(row, config):
    return row[config.selection_cost]


def choose(rows, config, *, representative=False):
    valid = [row for row in rows if row["status"] == "ok"]
    if config.selection_policy == "fastest_within_target":
        eligible = [row for row in valid if row["validation_score"] <= config.accuracy_target]
        if eligible:
            return min(eligible, key=lambda row: (cost(row, config), row["validation_score"]))
        if not representative:
            return None
    return min(valid, key=lambda row: row["validation_score"]) if valid else None


def describe(config):
    if config.selection_policy == "min_error":
        return f"Lowest validation {config.selection_metric}"
    return (f"Lowest measured {config.selection_cost} among methods with validation "
            f"{config.selection_metric} <= {config.accuracy_target:g}")
