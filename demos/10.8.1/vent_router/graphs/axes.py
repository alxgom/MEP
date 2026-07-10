from __future__ import annotations


def merge_close_values(values, threshold, preserve_values=None, priority_values=None):
    preserve_values = {round(float(v)) for v in (preserve_values or [])}
    priority_values = {round(float(v)) for v in (priority_values or [])}
    vals = sorted({round(float(v)) for v in values})
    if not vals:
        return []

    filtered = [vals[0]]
    for i in range(1, len(vals)):
        current = vals[i]
        if current in preserve_values:
            filtered.append(current)
        elif (
            i + 1 < len(vals)
            and abs(current - vals[i + 1]) < threshold
            and (vals[i + 1] in preserve_values or (current not in priority_values and vals[i + 1] in priority_values))
        ):
            continue
        elif abs(current - filtered[-1]) >= threshold:
            filtered.append(current)

    return filtered
