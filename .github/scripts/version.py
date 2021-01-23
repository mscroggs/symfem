import json
import sys
from datetime import datetime
from github import Github

failed = False

access_key = sys.argv[-2]
version = sys.argv[-1]

git = Github(access_key)

symfem = git.get_repo("mscroggs/symfem")
branch = symfem.get_branch("main")

vfile1 = symfem.get_contents("VERSION", branch.commit.sha)
v1 = vfile1.decoded_content.decode("utf8").strip()

latest_sha = branch.commit.sha

if v1 != version:
    latest_sha = symfem.update_file(
        "VERSION",
        "Update version number",
        version + "\n",
        sha=vfile1.sha,
        branch=branch.name,
    )["commit"]

    failed = True

vfile2 = symfem.get_contents("codemeta.json", branch.commit.sha)
data = json.loads(vfile2.decoded_content)
if data["version"] != version:
    data["version"] = version
    data["dateModified"] = datetime.now().strftime("%Y-%m-%d")
    latest_sha = symfem.update_file(
        "codemeta.json",
        "Update version number",
        json.dumps(data),
        sha=vfile2.sha,
        branch=branch_name,
    )["commit"]

    failed = True

if failed:
    rel = symfem.get_release(f"v{version}")
    rel.delete_release()

    symfem.create_git_tag_and_release(
        f"v{version}updated", "Version {version}", rel.title, rel.body,
        latest_sha, "commit")

    assert False
