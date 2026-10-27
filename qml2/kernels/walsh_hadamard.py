from ..jit_interfaces import bool_, jit_, ndarray_


@jit_
def unnormalized_fast_walsh_hadamard(array: ndarray_, forward: bool_ = True) -> None:
    """
    In-place unnormalized fast Walsh-Hadamard transform (natural/Hadamard order).

    The forward path uses radix-4 butterflies: each pass fuses two radix-2 levels
    (strides ``h`` and ``2h``) into a single sweep, so it makes half as many passes over
    the buffer as radix-2. The transform is memory-bandwidth bound, so pass count -- not
    flop count -- is what costs. When ``log2(n)`` is odd, one radix-2 level is done first
    to leave an even number of levels for the radix-4 loop.

    The unnormalized WHT is a symmetric involution (H == H.T), so ``forward`` does not
    affect the result; the reverse branch (currently unused) is kept as radix-2.
    """
    n = array.shape[0]
    if not forward:
        h = n // 2
        while h >= 1:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = array[j]
                    y = array[j + h]
                    array[j] = x + y
                    array[j + h] = x - y
            h //= 2
        return
    # Forward, radix-4. Number of levels = log2(n); do one radix-2 level first if it's odd.
    lg = 0
    m = n
    while m > 1:
        m >>= 1
        lg += 1
    h = 1
    if (lg & 1) == 1:
        for i in range(0, n, 2):
            x = array[i]
            y = array[i + 1]
            array[i] = x + y
            array[i + 1] = x - y
        h = 2
    while h < n:
        step = h * 4
        for i in range(0, n, step):
            for j in range(i, i + h):
                j1 = j + h
                j2 = j + 2 * h
                j3 = j + 3 * h
                x0 = array[j]
                x1 = array[j1]
                x2 = array[j2]
                x3 = array[j3]
                t0 = x0 + x1
                t1 = x0 - x1
                t2 = x2 + x3
                t3 = x2 - x3
                array[j] = t0 + t2
                array[j1] = t1 + t3
                array[j2] = t0 - t2
                array[j3] = t1 - t3
        h = step


@jit_
def generate_rf_stack_unbiased_phases(
    red_scaled_reps, temp_red_reps, sorf_diags, norm_const
) -> None:
    Ntransforms = sorf_diags.shape[0]
    if red_scaled_reps.shape[0] == temp_red_reps.shape[0]:
        temp_red_reps[:] = red_scaled_reps[:]
    else:
        size = red_scaled_reps.shape[0]
        temp_red_reps[:size] = red_scaled_reps[:]
        temp_red_reps[size:] = 0.0
    temp_red_reps *= norm_const ** (Ntransforms - 1)
    for transform in range(Ntransforms):
        temp_red_reps *= sorf_diags[transform]
        unnormalized_fast_walsh_hadamard(temp_red_reps)
