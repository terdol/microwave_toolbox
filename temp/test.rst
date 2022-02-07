Operating power gain from port1 to port2 with load impedance of ZL. If dB=True, output is in dB, otherwise it is a power ratio.
    .. math:: G_{op}=\\frac{P_{toLoad}}{P_{toNetwork}}
Args:
    port1 (int, optional): Index of input port. Defaults to 1.
    port2 (int, optional): Index of output port. Defaults to 2.
    ZL (ndarray or float, optional): Load impedance. Defaults to 50.0.
    dB (bool, optional): Enable dB output. Defaults to True.
Returns:
    numpy.ndarray: Array of Gop values for all frequencies