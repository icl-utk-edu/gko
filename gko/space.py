#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import skopt


class Space(skopt.space.Space):
    def __init__(self, dims):
        super(Space, self).__init__(dims)

    def __len__(self):
        return len(self.dims)

    def __getitem__(self, k):
        if isinstance(k, str):
            return super(Space, self)[k]

        elif isinstance(k, int):
            return self.dims[k]

        else:
            raise TypeError("Wrong subscript type (not str or int): "
                            "{}".format(type(k)))

    def __add__(self, other):
        return Space(self.dims + other.dims)

    def to_dict(self, plist):
        @skopt.utils.use_named_args(self.dims)
        def to_params_dict(**params):
            return params

        return to_params_dict(plist)


def factors(n):
    return skopt.space.Categorical(
        ["{},{}".format(f, n / f) for f in range(1, n + 1) if n % f == 0],
        transform="normalize", name="pq"
    )


def get_levels() -> list[Space]:
    blk = skopt.space.Integer(128, 2048, transform="normalize", name="blk")
    cast = skopt.space.Categorical(["a1", "a2"], transform="normalize",
                                   name="cast")
    powers = [2**i for i in range(5)]
    return [Space([blk, factors(powers[i] * 6), cast])
            for i in range(len(powers))]
