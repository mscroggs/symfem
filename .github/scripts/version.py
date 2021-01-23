import json
import sys
from datetime import datetime
import github

failed = False

access_key = sys.argv[-2]
version = sys.argv[-1]

assert "updated" not in version
assert version[0] == "v"
version = version[1:]

git = github.Github(access_key)

symfem = git.get_repo("mscroggs/symfem")
branch = symfem.get_branch("main")
ref = symfem.get_git_ref("heads/main")
base_tree = symfem.get_git_tree(branch.commit.sha)

vfile1 = symfem.get_contents("VERSION", branch.commit.sha)
v1 = vfile1.decoded_content.decode("utf8").strip()

changed_list = []

if v1 != version:
    element = github.InputGitTreeElement("VERSION", '100644', 'blob', f"{version}\n")
    changed_list.append(element)
    failed = True

vfile2 = symfem.get_contents("codemeta.json", branch.commit.sha)
data = json.loads(vfile2.decoded_content)
if data["version"] != version:
    data["version"] = version
    data["dateModified"] = datetime.now().strftime("%Y-%m-%d")
    element = github.InputGitTreeElement("codemeta.json", '100644', 'blob', json.dumps(data))
    changed_list.append(element)
    failed = True


if failed:
    tree = symfem.create_git_tree(changed_list, base_tree)
    parent = symfem.get_git_commit(branch.commit.sha)
    commit = symfem.create_git_commit("Update version numbers", tree, [parent])
    ref.edit(commit.sha)

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
        f"v{version}", title, title, body, commit.sha, "commit")

    assert False
