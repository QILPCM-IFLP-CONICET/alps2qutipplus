def split_nbody_terms(operator: NBodyOperator):
    """Returns a dictionary with the terms collected by the
    number of sites over which the term acts"""
    result_dict = {}
    for term in operator.terms:
        if isinstance(term, NBodyOperator):
            for n, t in split_nbody_terms(term).items():
                result_dict[n] = result_dict.get(n, 0) + t
            continue
        if isinstance(term, LocalOperator):
            result_dict[1] = result_dict.get(1, 0) + term
        if isinstance(term, ProductOperator):
            n = len(term.sites_op)
            result_dict[n] = result_dict.get(n, 0) + term

    return result_dict
