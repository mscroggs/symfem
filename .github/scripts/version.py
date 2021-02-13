import json
import sys
from datetime import datetime
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




failed = False


if failed:
    assert not is_update
    tree = symfem.create_git_tree(changed_list, base_tree)
    parent = symfem.get_git_commit(branch.commit.sha)
    commit = symfem.create_git_commit("Update version numbers", tree, [parent])
    ref.edit(commit.sha, force=True)

    try:
        rel = symfem.get_release(f"v{version}")
        rel.delete_release()
        title = rel.title
        body = rel.body
    except:  # noqa: E722
        print("No release found")
        title = f"Version {version}"
        body = ""

    symfem.create_git_tag_and_release(
        f"v{version}", f"Version {version}", f"Version {version}", "Latest release",
        branch.commit.sha, "commit")
