import json
import sys
import github
import urllib.request

access_key = sys.argv[-1]

git = github.Github(access_key)

symfem = git.get_repo("mscroggs/symfem")
branch = symfem.get_branch("main")

version = symfem.get_contents("VERSION", branch.commit.sha).decoded_content.decode()

print(f"https://pypi.org/pypi/symfem/{version}/json")
with urllib.request.urlopen(f"https://pypi.org/pypi/symfem/{version}/json") as f:
    data = json.load(f)

for file in data["releases"][version]:
    if file["packagetype"] == "sdist":
        hash = file["digests"]["sha256"]

print(hash)
