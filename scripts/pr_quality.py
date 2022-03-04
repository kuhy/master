#!/usr/bin/env python3
import click
import datetime
import git
import json
import os
import shutil
import sys
import subprocess
import tempfile
import time


def obtain_project_data(user: str, repo: str, db_name: str, db_user: str,
                        db_password: str):
    output = subprocess.check_output(["python", "gh_info.py", "-d", db_name,
                                      "-u", db_user, "-p", db_password,
                                      user, repo])
    return json.loads(output)


def make_commit_accesible(repo, remote, commit_sha):
    try:
        repo.commit(commit_sha)
        return True
    except ValueError:
        if remote not in repo.remotes:
            repo.create_remote(remote,
                               url="https://null:null@github.com/" + remote +
                               ".git")
            try:
                repo.remotes[remote].fetch()
                print("Remote " + remote + " was successfully added.")
                return make_commit_accesible(repo, remote, commit_sha)
            except git.exc.GitError:
                print("Repository " + remote + " is not accessible.")
                repo.delete_remote(remote)
                return False
        else:
            print("Commit {} is not present in the {} repository."
                  .format(commit_sha, remote))
            return False


def analyze_project(project_data, tmp_dir):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          " Starting analyzing project " + project_data["name"] + ".")
    start_time = time.time()
    project_dir = tmp_dir + "/" + project_data["name"]
    repo = git.Repo.init(project_dir)
    repo.create_remote("origin", url="https://github.com/" +
                       project_data["name"] + ".git")
    repo.remotes.origin.fetch()
    print("Project " + project_data["name"] + " was cloned into the " +
          project_dir + " directory.")
    pull_requests = []
    for pr_data in project_data["pull_requests"]:
        try:
            repo.commit(pr_data["base_commit"])
        except ValueError:
            print("Base commit {} is not present in the repository."
                  .format(pr_data["base_commit"]))
            continue
        if not make_commit_accesible(repo, pr_data["head_repo"],
                                     pr_data["head_commit"]):
            print("Skipping pull request no. " + str(pr_data["id"]) + ".")
            continue
        output = subprocess.check_output(["git-contrast",
                                          "--output-format=json",
                                          pr_data["base_commit"],
                                          pr_data["head_commit"]
                                          ], cwd=project_dir)
        pr_data.update(json.loads(output))
        pull_requests.append(pr_data)
    project_data["pull_requests"] = pull_requests
    print("Removing project directory: " + project_dir)
    shutil.rmtree(project_dir)
    print("Time to analyze project " + project_data["name"] + ": " +
          str(time.time() - start_time) + "\n")


@click.option("--db-name", "-d", required=True)
@click.option("--db-user", "-u", required=True)
@click.option("--db-password", "-p", required=True, prompt="Database password")
@click.option("--tmp-dir-path", "-t")
@click.command()
def cli(db_name: str, db_user: str, db_password: str,
        tmp_dir_path: str = None):
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_path)
    for line in sys.stdin.readlines():
        project_data = obtain_project_data(*line.rstrip().split("/"), db_name,
                                           db_user, db_password)
        analyze_project(project_data, tmp_dir)
        filename = "data/" + project_data["name"] + ".json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(json.dumps(project_data))


if __name__ == "__main__":
    cli(auto_envvar_prefix="PR_QUALITY")
