"""Backward compatible entry point for :mod:`glucoalg.evaluation`."""


def __getattr__(name):
    from glucoalg import evaluation

    return getattr(evaluation, name)


if __name__ == "__main__":
    from glucoalg.eval_grid import legacy_main

    legacy_main()
