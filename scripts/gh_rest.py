#!/usr/bin/env python3
"""Obtain data about given repository using the GitHub REST API.

WARNING: The user needs to provide GitHub API token.
"""

import click
import datetime
import json
import requests


GH_URL = "https://api.github.com"


def obtain_project_data(project_name: str, gh_user: str, gh_token: str):
    """Obtain data about project using the GitHub REST API."""
    data = json.loads(requests.get(f"{GH_URL}/repos/{project_name}",
                                   auth=(gh_user, gh_token)).text)
    project_data = {}
    project_data["name"] = data["full_name"]
    project_data["language"] = data["language"]
    project_data["created_at"] = datetime.datetime.strptime(
        data["created_at"], '%Y-%m-%dT%H:%M:%SZ').timestamp()
    project_data["number_of_forks"] = data["forks"]
    project_data["number_of_commits"] = None
    project_data["number_of_project_members"] = None
    project_data["number_of_watchers"] = data["stargazers_count"]
    project_data["pull_requests"] = obtain_prs_data(project_name, gh_user,
                                                    gh_token)
    return project_data


def obtain_prs_data(project_name: str, gh_user: str, gh_token: str):
    """Obtain data about project pull requests."""
    prs_data = []
    i = 1
    while True:
        data = json.loads(requests.get(f"{GH_URL}/repos/{project_name}/pulls?"
                                       f"state=closed&per_page=100&page={i}",
                                       auth=(gh_user, gh_token)).text)
        if len(data) == 0:
            break
        for pr in data:
            pr_data = {}
            pr_data["id"] = pr["number"]
            pr_data["accepted"] = pr["merged_at"] is not None
            pr_data["time_opened"] = (
                datetime.datetime.strptime(pr["closed_at"],
                                           '%Y-%m-%dT%H:%M:%SZ').timestamp() -
                datetime.datetime.strptime(pr["created_at"],
                                           '%Y-%m-%dT%H:%M:%SZ').timestamp())
            pr_data["head_repo"] = (pr["head"]["repo"]["full_name"]
                                    if pr["head"]["repo"] else None)
            pr_data["head_commit"] = pr["head"]["sha"]
            pr_data["base_commit"] = pr["base"]["sha"]
            pr_data["number_of_commits"] = None
            pr_data["number_of_comments"] = None
            pr_data["submitter"] = {"username": pr["user"]["login"],
                                    "number_of_followers": None,
                                    "is_project_member":
                                    pr["author_association"] == "MEMBER"}
            prs_data.append(pr_data)
        i += 1
    return prs_data


@click.command()
@click.argument("user")
@click.argument("repo")
@click.option("--gh-user", "-u", required=True)
@click.option("--gh-token", "-t", required=True)
def cli(user: str, repo: str, gh_user: str, gh_token: str):
    """Obtain data about given repository using the GitHub REST API."""
    project_data = obtain_project_data(user + '/' + repo, gh_user, gh_token)
    print(json.dumps(project_data))


if __name__ == "__main__":
    cli(auto_envvar_prefix="GH_REST")
