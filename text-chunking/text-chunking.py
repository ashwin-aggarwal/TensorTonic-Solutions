def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    # Write code here
    step_size = chunk_size - overlap

    output = []

    n = len(tokens)

    for i in range(0,n, step_size):
        output.append(tokens[i:i + chunk_size])
        if i+chunk_size >= len(tokens):
            break
        

    return output
        