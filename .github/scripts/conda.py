import json
import sys
import github
import urllib.request

access_key = sys.argv[-1]

git = github.Github(access_key)

symfem = git.get_repo("mscroggs/symfem")
branch = symfem.get_branch("main")

version = symfem.get_contents("VERSION", branch.commit.sha).decoded_content.decode()

with urllib.request.urlopen(f"https://pypi.org/pypi/symfem/{version}/json") as f:
    data = json.load(f)

for file in data["releases"][version]:
    if file["packagetype"] == "sdist":
        hash = file["digests"]["sha256"]

upstream_feedstock = git.get_repo("conda-forge/symfem-feedstock")
upstream_branch = upstream_feedstock.get_branch("master")

fork = git.get_user().create_fork(upstream_feedstock)

u = git.get_user()

for repo in u.get_repos():
    if repo.full_name.startswith("symfembot/symfem-feedstock"):
        repo.delete()

fork = git.get_user().create_fork(upstream_feedstock)
branch = fork.get_branch("master")

old_meta = fork.get_contents("recipe/meta.yaml", branch.commit.sha)

old_meta_lines = old_meta.decoded_content.decode().split("\n")
new_meta_lines = []
for line in old_meta_lines:
    if line.startswith("{% set version"):
        new_meta_lines.append(f"{{% set version = \"{version}\" %}}")
    elif "sha256" in line:
        newline = line.split("sha256")[0]
        newline += f"sha256: {hash}"
        new_meta_lines.append(newline)
    elif "number" in line:
        newline = line.split("number")[0]
        newline += "number: 0"
        new_meta_lines.append(newline)
    else:
        new_meta_lines.append(line)

fork.update_file("recipe/meta.yaml", "Update version", "\n".join(new_meta_lines), sha=old_meta.sha)

upstream_feedstock.create_pull(title=f"Update version to {version}",
                               body="", base="master", head="symfembot:master")
