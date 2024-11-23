def extract(a, t, x_shape):
    """
    Extracts values from tensor `a` at indices specified by tensor `t` and reshapes the result.

    Args:
        a (torch.Tensor): The input tensor from which values are to be extracted.
        t (torch.Tensor): A tensor containing the indices at which to extract values from `a`.
        x_shape (tuple): The shape of the output tensor.

    Returns:
        torch.Tensor: A tensor containing the extracted values, reshaped to match `x_shape`.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
