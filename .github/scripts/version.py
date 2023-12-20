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

for release in symfem.get_releases():
    if release.tag_name == f"v{version}":
        print("release=no")
        break
else:
    print(f"release={version}")
    symfem.create_git_ref(ref=f"refs/heads/v{version}-changelog", sha=branch.commit.sha)
    new_branch = symfem.get_branch(f"v{version}-changelog")
    changelog_file = symfem.get_contents("CHANGELOG_SINCE_LAST_VERSION.md", new_branch.commit.sha)
    changes = changelog_file.decoded_content.decode("utf8").strip()

    if changes == "":
        raise RuntimeError("CHANGELOG_SINCE_LAST_VERSION.md should not be empty")

    symfem.create_git_tag_and_release(
        f"v{version}", f"Version {version}", f"Version {version}", changes,
        branch.commit.sha, "commit")

    old_changelog_file = symfem.get_contents("CHANGELOG.md", new_branch.commit.sha)
    old_changes = old_changelog_file.decoded_content.decode("utf8").strip()

    new_changelog = (f"# Version {version} ({datetime.now().strftime('%d %B %Y')})\n\n"
                     f"{changes}\n\n{old_changes}\n")

    symfem.update_file(
        "CHANGELOG.md", "Update CHANGELOG.md", new_changelog, sha=old_changelog_file.sha,
        branch=f"v{version}-changelog"
    )
    symfem.update_file(
        "CHANGELOG_SINCE_LAST_VERSION.md", "Reset CHANGELOG_SINCE_LAST_VERSION.md", "",
        sha=changelog_file.sha, branch=f"v{version}-changelog"
    )

    symfem.create_pull(
        title="Update changelogs", body="", base="main", head=f"v{version}-changelog"
    )
