import json
import sys
import github

access_key = sys.argv[-1]

git = github.Github(access_key)

symfem = git.get_repo("mscroggs/symfem")
branch = symfem.get_branch("main")
ref = symfem.get_git_ref("heads/main")
base_tree = symfem.get_git_tree(branch.commit.sha)

vfile1 = symfem.get_contents("VERSION", branch.commit.sha)
version = vfile1.decoded_content.decode("utf8").strip()

vfile2 = symfem.get_contents("codemeta.json", branch.commit.sha)
data = json.loads(vfile2.decoded_content)
assert data["version"] == version

release = symfem.get_releases()[0].tag_name

if release != f"v{version}":
    symfem.create_git_tag_and_release(
        f"v{version}", f"Version {version}", f"Version {version}", "Latest release",
        branch.commit.sha, "commit")
