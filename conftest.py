# Intentionally has no fixtures. Its presence makes pytest put the
# project root on sys.path, so `import seacode` works when running
# `pytest` before (or without) `pip install -e .`. Deleting it breaks
# test collection in that case.
