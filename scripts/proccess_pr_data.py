#!/usr/bin/env python3
import json
import sys


data = json.loads(sys.stdin.read())
output = {"total": []}

for i, pr in enumerate(data["pull_requests"]):
    for key, value in pr.items():
        if key == "results":
            output["total"].append(0)
            for linter, results in value.items():
                for issue, issue_info in results.items():
                    issue_name = linter + "_" + issue
                    diff = issue_info["post"] - issue_info["pre"]
                    if issue_name not in output:
                        output[issue_name] = [0] * i
                    output[issue_name].append(diff)
                    output["total"][-1] += diff
            for key, value in output.items():
                if len(value) != i + 1:
                    output[key].extend([0] * (i + 1 - len(value)))
        elif key == "submitter":
            for submitter_key, submitter_value in value.items():
                out_key = "submitter_" + submitter_key
                if out_key not in output:
                    output[out_key] = []
                output[out_key].append(submitter_value)
        else:
            if key not in output:
                output[key] = []
            output[key].append(value)

number_of_pull_requests = len(data["pull_requests"])
for key, value in data.items():
    if key != "pull_requests":
        output["project_" + key] = [value] * number_of_pull_requests

print(json.dumps(output))
