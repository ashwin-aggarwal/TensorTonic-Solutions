def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    # Write code here
    res, prev = [] , series[0]

    for step in series[1:]:
        if prev == 0:
            prev = (0.0)
            res.append(prev)
        else:
            change = (step - prev) / prev
            res.append(change)
        prev = step

    return res