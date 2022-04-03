#!/usr/bin/env python3
"""
Transform projects data into CSV files.

This little script reads project data from JSON files on the given path and
writes the filtered information about pull requests into CSV files.
"""

import click
import csv
import json
import os
import pathlib


def filter_pull_requests(project):
    """Remove pull request that do not change any source code files."""
    main_linter = {"C": "OCLint", "C++": "OCLint", "Haskell": "HLint",
                   "Java": "PMD", "Kotlin": "ktlint", "Python": "Pylint"
                   }[project["language"]]

    # keep only pull requests that change some source code of the main language
    project["pull_requests"] = [pr for pr in project["pull_requests"]
                                if any(key in pr for key in
                                       [f"modified_{main_linter}",
                                        f"added_{main_linter}",
                                        f"deleted_{main_linter}"])]

    # keep only results from the main linter
    for pr in project["pull_requests"]:
        pr["results"] = {linter: linter_results for (linter, linter_results)
                         in pr["results"].items() if linter == main_linter}

    # remove pull requests where parsing error occurred
    if project["language"] == "Haskell":
        project["pull_requests"] = [pr for pr in project["pull_requests"]
                                    if len(pr["results"]) == 0 or
                                    not any(issue.startswith("Parse-error")
                                            for issue
                                            in pr["results"]["HLint"].keys())]


def project_to_pull_requests(project):
    """Convert project data to list of pull requests (list of dictionaries)."""
    project = project.copy()
    prs_flat = []
    prs = project.pop("pull_requests")
    project = {f"project_{key}": val for key, val in project.items()}
    for pr in prs:
        submitter = {f"submitter_{key}": val for key, val
                     in pr.pop("submitter").items()}
        results = {f"results_{linter}_{issue_results['type']}_{issue}":
                   issue_results["post"] - issue_results["pre"]
                   for (linter, linter_results) in pr.pop("results").items()
                   for (issue, issue_results) in linter_results.items()}
        prs_flat.append(project | submitter | pr | results)
    return prs_flat


def lod_to_dol(list_of_dictionaries):
    """
    Convert list of dictionaries to dictionary of lists.

    Resulting lists will have the same size.
    Missing items are replaced with zeros.
    """
    dictionary_of_lists = {}
    for i, pr in enumerate(list_of_dictionaries):
        for key, val in pr.items():
            if key not in dictionary_of_lists:
                dictionary_of_lists[key] = [0] * i
            dictionary_of_lists[key].append(val)
        for key, val in dictionary_of_lists.items():
            if len(val) < i + 1:
                dictionary_of_lists[key].append(0)
    return dictionary_of_lists


def save_dol_as_csv(dictionary_of_lists, csv_file_path):
    """Save dictionary of lists into the CSV file on the given path."""
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    with open(csv_file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(dictionary_of_lists.keys())
        csv_writer.writerows(zip(*dictionary_of_lists.values()))


@click.command()
@click.argument("path")
def cli(path: str):
    """Transform projects data into CSV files."""
    language_prs = {}
    for project_file_path in pathlib.Path(path).rglob("**/*.json"):
        with open(project_file_path, 'r') as f:
            project = json.loads(f.read())
        filter_pull_requests(project)
        prs = project_to_pull_requests(project)
        if project["language"] not in language_prs:
            language_prs[project["language"]] = []
        language_prs[project["language"]].extend(prs)
        save_dol_as_csv(lod_to_dol(prs), "csv/{}/{}.csv".format(
            project["language"], project["name"].split("/")[1]))
    for language in language_prs.keys():
        save_dol_as_csv(lod_to_dol(language_prs[language]),
                        f"csv/{language}.csv")


if __name__ == "__main__":
    cli()
