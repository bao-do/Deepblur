def grad(input, dim=3):

    if isinstance(dim, (list, tuple)):
        return [grad(input, d) for d in dim]

    gindex = [slice(None)] * input.dim()
    gindex[dim] = slice(1, None)

    lindex = [slice(None)] * input.dim()
    lindex[dim] = slice(None, -1)
    return input[tuple(gindex)] - input[tuple(lindex)]

def as_pair(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return (x, x)