with open(".coveralls_output") as f:
    assert int(f.read().split("TOTAL")[-1].split("%")[0].split(" ")[-1]) > 80
