import json
from datetime import datetime

now = datetime.now()

with open("VERSION") as f:
    version = tuple(int(i) for i in f.read().split("."))

if now.year == version[0] and now.month == version[1]:
    new_version = (now.year, now.month, version[2] + 1)
else:
    new_version = (now.year, now.month, 1)

new_version_str = ".".join([f"{i}" for i in new_version])

with open("VERSION", "w") as f:
    f.write(new_version_str)

with open("codemeta.json") as f:
    data = json.load(f)
data["version"] = new_version_str
data["dateModified"] = now.strftime("%Y-%m-%d")
with open("codemeta.json", "w") as f:
    json.dump(data, f)

print(f"Updated version to {new_version_str}")

