"""``taxel doctor``: the integrity check and the five health checks on a recorded dataset.

The check functions take numpy arrays and return report blocks; they are pure and need
numpy only. :mod:`taxel.doctor.loader` turns a dataset root into those arrays, reading
parquet with pyarrow (the ``taxel[doctor]`` extra) and video through LeRobot
(``taxel[lerobot]``), both imported where they are used.
"""
