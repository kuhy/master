#!/usr/bin/env -S python3 -u
import click
import datetime
import functools
import git
import json
import os
import shutil
import sys
import subprocess
import tempfile
import time


def obtain_project_data_from_database(db_name: str, db_user: str,
                                      db_password: str, user: str, repo: str):
    print(f"Obtaining information about {user}/{repo} via GHTorrent database.")
    output = subprocess.check_output(["python", "gh_db.py", "-d", db_name,
                                      "-u", db_user, "-p", db_password,
                                      user, repo])
    return json.loads(output)


def obtain_project_data_from_rest(gh_user: str, gh_token: str, user: str,
                                  repo: str):
    print(f"Obtaining information about {user}/{repo} via GitHub REST API.")
    output = subprocess.check_output(["python", "gh_rest.py", "-u", gh_user,
                                      "-t", gh_token, user, repo])
    return json.loads(output)


def make_commit_accesible(repo, remote, commit_sha):
    try:
        repo.commit(commit_sha)
        return True
    except ValueError:
        if remote is None:
            print("Information about the head repository is not available.")
            return False
        elif remote not in repo.remotes:
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
    number_of_usable_prs = 0
    for pr_data in project_data["pull_requests"]:
        try:
            repo.commit(pr_data["base_commit"])
        except ValueError:
            print("Base commit {} is not present in the repository."
                  .format(pr_data["base_commit"]))
            print("Skipping pull request no. " + str(pr_data["id"]) + ".")
            continue
        if not make_commit_accesible(repo, pr_data["head_repo"],
                                     pr_data["head_commit"]):
            print("Skipping pull request no. " + str(pr_data["id"]) + ".")
            continue
        try:
            print("Linting pull request no. " + str(pr_data["id"]) + ".")
            output = subprocess.check_output(["git-contrast",
                                              "--output-format=json",
                                              "--language",
                                              project_data["language"],
                                              pr_data["base_commit"],
                                              pr_data["head_commit"]
                                              ], cwd=project_dir,
                                             timeout=1000)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print("Execution of git-contrast failed:")
            print(e)
            print("Skipping pull request no. " + str(pr_data["id"]) + ".")
            continue
        pr_data.update(json.loads(output))
        pull_requests.append(pr_data)
        if any([key.startswith(("modified_", "added_", "deleted_")) for key in
                pr_data.keys()]):
            number_of_usable_prs += 1
            if number_of_usable_prs == 500:
                print("Skipping rest of the pull requests. "
                      "(sufficient number of pull requests)")
                break
            elif number_of_usable_prs % 10 == 0:
                project_data["pull_requests"] = pull_requests
                filename = "interim/" + project_data["name"] + ".json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                print("Backing up the project data into " + filename)
                with open(filename, "w") as f:
                    f.write(json.dumps(project_data))
        else:
            print("Pull request no. " + str(pr_data["id"]) + " is not usable.")
    project_data["pull_requests"] = pull_requests
    print("Removing project directory: " + project_dir)
    shutil.rmtree(project_dir)
    if number_of_usable_prs < 200:
        print("WARNING: Number of usable pull requests is small: " +
              str(number_of_usable_prs))
    print("Time to analyze project " + project_data["name"] + ": " +
          str(time.time() - start_time))


@click.group()
@click.option("--tmp-dir-path", "-t",
              type=click.Path(exists=True, file_okay=False))
@click.pass_context
def cli(ctx, tmp_dir_path: str = None):
    ctx.ensure_object(dict)
    ctx.obj["tmp_dir_path"] = tmp_dir_path


@cli.command()
@click.option("--db-name", "-d", required=True)
@click.option("--db-user", "-u", required=True)
@click.option("--db-password", "-p", required=True, prompt="Database password")
@click.pass_context
def db(ctx, db_name: str, db_user: str, db_password: str):
    analyze_projects(functools.partial(obtain_project_data_from_database,
                                       db_name, db_user, db_password),
                     ctx.obj["tmp_dir_path"])


@cli.command()
@click.option("--gh-user", "-u", required=True)
@click.option("--gh-token", "-t", required=True)
@click.pass_context
def rest(ctx, gh_user: str, gh_token: str):
    analyze_projects(functools.partial(obtain_project_data_from_rest, gh_user,
                                       gh_token), ctx.obj["tmp_dir_path"])


def analyze_projects(obtain_project_data, tmp_dir_path: str = None):
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_path)
    for line in sys.stdin.readlines():
        project_data = obtain_project_data(*line.rstrip().split("/"))
        analyze_project(project_data, tmp_dir)
        filename = "data/" + project_data["name"] + ".json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print("Saving the results into " + filename + "\n")
        with open(filename, "w") as f:
            f.write(json.dumps(project_data))


if __name__ == "__main__":
    cli(obj={}, auto_envvar_prefix="PR_QUALITY")
