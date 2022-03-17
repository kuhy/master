#!/usr/bin/env python3
import click
import json
import mysql.connector
import time


def obtain_project_data(project_name: str, db):
    project_data = {}
    project_data["name"] = project_name

    cur = db.cursor()
    cur.execute("SELECT id, language, created_at FROM projects WHERE url = %s LIMIT 1",
                ("https://api.github.com/repos/" + project_name,))
    (project_id, language, created_at) = cur.fetchone()
    project_data["language"] = language
    project_data["created_at"] = time.mktime(created_at.timetuple())

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM projects WHERE forked_from = %s",
                (project_id,))
    project_data["number_of_forks"] = cur.fetchone()[0]

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM project_commits WHERE project_id = %s",
                (project_id,))
    project_data["number_of_commits"] = cur.fetchone()[0]

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM project_members WHERE repo_id = %s",
                (project_id,))
    project_data["number_of_project_members"] = cur.fetchone()[0]

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM watchers WHERE repo_id = %s",
                (project_id,))
    project_data["number_of_watchers"] = cur.fetchone()[0]

    cur = db.cursor()
    cur.execute("SELECT id, head_repo_id, head_commit_id, base_repo_id, "
                "base_commit_id, pullreq_id FROM pull_requests WHERE "
                "base_repo_id = %s",
                (project_id,))
    prs = []
    for pr in cur:
        prs.append(pr)
    project_data["pull_requests"] = []
    for pr in prs:
        pr_data = obtain_pr_data(*pr, db)
        if pr_data:
            project_data["pull_requests"].append(pr_data)

    return project_data


def obtain_pr_data(pr_id: int, head_repo_id: int, head_commit_id: int,
                   base_repo_id: int, base_commit_id: int, github_pr_id:
                   int, db):
    if not head_repo_id:
        return  # head repository was deleted

    opened_at = None
    is_merged = False
    is_closed = False
    cur = db.cursor()
    cur.execute("SELECT action, created_at, actor_id FROM pull_request_history"
                " WHERE pull_request_id = %s ORDER BY created_at", (pr_id,))
    for (action, created_at, actor_id) in cur:
        if action == "opened":
            opened_at = time.mktime(created_at.timetuple())
            submitter_id = actor_id
        elif action == "closed":
            is_closed = True
            closed_at = time.mktime(created_at.timetuple())
        elif action == "merged":
            is_merged = True
        elif action == "reopened":
            is_closed = False

    if not is_closed:
        return  # PR is not closed

    pr_data = {}

    pr_data["id"] = github_pr_id

    pr_data["accepted"] = is_merged

    if not opened_at:
        return  # incomplete data in the DB

    pr_data["time_opened"] = closed_at - opened_at

    cur = db.cursor()
    cur.execute("SELECT url FROM projects WHERE id = %s", (head_repo_id,))
    url = cur.fetchone()[0]
    if not url.startswith("https://api.github.com/repos/"):
        raise ValueError("Unexpected format of repo URL: " + url)
    pr_data["head_repo"] = url[len("https://api.github.com/repos/"):]

    cur = db.cursor()
    cur.execute("SELECT sha FROM commits WHERE id = %s", (head_commit_id,))
    head_commit_sha = cur.fetchone()
    if not head_commit_sha:
        return  # head commit was deleted
    pr_data["head_commit"] = head_commit_sha[0]

    cur = db.cursor()
    cur.execute("SELECT sha FROM commits WHERE id = %s", (base_commit_id,))
    pr_data["base_commit"] = cur.fetchone()[0]

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM pull_request_commits WHERE "
                "pull_request_id = %s", (pr_id,))
    pr_data["number_of_commits"] = cur.fetchone()[0]

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM pull_request_comments WHERE "
                "pull_request_id = %s", (pr_id,))
    pr_data["number_of_comments"] = cur.fetchone()[0]

    if not submitter_id:
        return  # submitter account was deleted

    pr_data["submitter"] = obtain_submitter_data(submitter_id, base_repo_id,
                                                 db)

    return pr_data


def obtain_submitter_data(submitter_id: int, project_id: int, db):
    submitter_data = {}

    cur = db.cursor()
    cur.execute("SELECT login FROM users WHERE id = %s", (submitter_id,))
    submitter_data["username"] = cur.fetchone()[0]

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM followers WHERE user_id = %s",
                (submitter_id,))
    submitter_data["number_of_followers"] = cur.fetchone()[0]

    cur = db.cursor()
    cur.execute("SELECT * FROM project_members WHERE repo_id = %s AND "
                "user_id = %s", (project_id, submitter_id))
    submitter_data["is_project_member"] = len(cur.fetchall()) != 0

    return submitter_data


@click.command()
@click.argument("user")
@click.argument("repo")
@click.option("--db-name", "-d", required=True)
@click.option("--db-user", "-u", required=True)
@click.option("--db-password", "-p", required=True, prompt="Database password")
def cli(user: str, repo: str, db_name: str, db_user: str, db_password: str):
    db = mysql.connector.connect(user=db_user, password=db_password,
                                 database=db_name)
    project_data = obtain_project_data(user + '/' + repo, db)
    print(json.dumps(project_data))
    db.close()


if __name__ == "__main__":
    cli(auto_envvar_prefix="GH_INFO")
