# scripts/review_pull_request.py
# Purpose: Validate Pull Request (PR) rules according to project conventions.

import os
import re
import sys
from typing import Dict, Any, Tuple

# Centralized imports
from git import get_pr_context

from ui import log_message, Colors, STATUS_COLORS  # Use shared UI module

# --- CONFIGURATION ---
REPO_ROOT = os.getcwd()
MAIN_BRANCHES = ['master', 'develop', 'staging']
WORK_BRANCHES = ['feature', 'fix', 'bugfix', 'hotfix', 'release']
ALLOWED_BRANCH_TYPES = MAIN_BRANCHES + WORK_BRANCHES
CONVENTIONAL_COMMIT_TYPES = {'feat', 'fix', 'build', 'chore', 'ci', 'docs', 'style', 'refactor', 'perf', 'test'}
CRITICAL_SEVERITY = 'Alta'
MEDIUM_SEVERITY = 'Media'
LOW_SEVERITY = 'Baja'
MAX_FAILURES_ALLOWED = 4


def get_context() -> Dict[str, Any]:
    """
    Gets and enriches the Pull Request context.
    Retrieves the base context from Git and adds version information
    extracted from `README.md` and `.VERSION` files if they exist.
    """
    context = get_pr_context()

    # Extract version from Changelog in README.md
    readme_path = os.path.join(REPO_ROOT, 'README.md')
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                changelog_section = re.search(r"##\s*Changelog.*", content, re.IGNORECASE)
                if changelog_section:
                    changelog_content = content[changelog_section.end():]
                    version_match = re.search(r"v(\d+\.\d+\.\d+)", changelog_content, re.IGNORECASE)
                    if version_match:
                        context['version_from_changelog'] = version_match.group(1)
        except IOError as e:
            log_message(f"Could not read {readme_path}: {e}", "WARNING")

    # Extract version from .VERSION file
    version_file_path = os.path.join(REPO_ROOT, '.VERSION')
    if os.path.exists(version_file_path):
        try:
            with open(version_file_path, 'r', encoding='utf-8') as f:
                context['version_from_file'] = f.read().strip()
        except IOError as e:
            log_message(f"Could not read {version_file_path}: {e}", "WARNING")

    return context


def check_branch_type(branch_type: str) -> Tuple[bool, str]:
    """Validates if a branch type is allowed."""
    if branch_type in ALLOWED_BRANCH_TYPES:
        return True, f"Branch type '{branch_type}' is allowed."
    return False, f"Branch type '{branch_type}' is not allowed. Allowed: {ALLOWED_BRANCH_TYPES}"


def check_source_branch_type(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates the type of the PR's source branch."""
    branch_type = context.get('branch_name', '').split('/')[0]
    return check_branch_type(branch_type)


def check_destination_branch_type(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates the type of the PR's destination branch."""
    branch_type = context.get('bitbucket_pr_destination_branch', '').split('/')[0]
    return check_branch_type(branch_type)


def check_branch_name_structure(branch: str) -> Tuple[bool, str]:
    """Validates the structure of a branch name."""
    if not branch:
        return False, "Branch name not available."

    parts = branch.split('/')
    branch_type = parts[0]

    if branch_type in MAIN_BRANCHES:
        return True, f"Main branch name '{branch}' is valid."

    if branch_type in WORK_BRANCHES:
        if len(parts) > 1:
            name_part = '/'.join(parts[1:])
            # Structure: type/TICKET-description
            if re.match(r'^[A-Z0-9]+[0-9]{2}-[0-9]+-.*', name_part):
                return True, f"Branch name '{branch}' follows the required structure."
        return False, f"Work branch name '{branch}' does not follow the 'type/TICKET-description' structure."

    return False, f"Unknown branch type '{branch_type}' in '{branch}'."


def check_source_branch_name(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates the structure of the source branch name."""
    return check_branch_name_structure(context.get('branch_name', ''))


def check_destination_branch_name(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates the structure of the destination branch name."""
    return check_branch_name_structure(context.get('bitbucket_pr_destination_branch', ''))


def check_readme_changelog_entry(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates that README.md has been modified with a changelog entry."""
    if 'README.md' not in context.get('modified_files', []):
        return False, "README.md was not modified."

    if context.get('version_from_changelog'):
        return True, f"Changelog entry found (v{context['version_from_changelog']})."

    return False, "No changelog entry with version format (vX.Y.Z) was found."


def check_version_consistency(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates that versions in .VERSION and README.md are consistent."""
    if '.VERSION' not in context.get('modified_files', []):
        return False, ".VERSION was not modified."

    version_file = context.get('version_from_file')
    version_changelog = context.get('version_from_changelog')

    if not version_file or not version_changelog:
        return False, "Version is missing from .VERSION or the README changelog."

    if not re.match(r'^\d+\.\d+\.\d+$', version_file):
        return False, f"Version '{version_file}' in .VERSION is not semantic (must be X.Y.Z)."

    if version_file == version_changelog:
        return True, f"Version {version_file} is consistent in .VERSION and changelog."

    return False, f"Version inconsistency: .VERSION='{version_file}', changelog='v{version_changelog}'."


def check_conventional_commits(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates that commit messages follow the Conventional Commits standard."""
    pattern = re.compile(r"^(\w+)(?:\((.+)\))?(!?):\s(.+)")
    non_compliant_commits = []

    for commit_msg in context.get('commit_messages', []):
        if commit_msg.lower().startswith(("merge", "revert")):
            continue

        match = pattern.match(commit_msg)
        if not match:
            non_compliant_commits.append(f"Invalid structure -> '{commit_msg}'")
            continue

        commit_type = match.group(1)
        if commit_type not in CONVENTIONAL_COMMIT_TYPES:
            non_compliant_commits.append(f"Type '{commit_type}' not allowed -> '{commit_msg}'")

    if not non_compliant_commits:
        return True, "All commits comply with the Conventional Commits standard."

    return False, f"Non-compliant commits: {'; '.join(non_compliant_commits)}"


def check_repo_name(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates the repository name against a defined pattern."""
    repo_name = context.get('bitbucket_repo_name', '')
    pattern = r'^([a-zA-Z0-9]+-){2,3}(ios|android|svc|webapp|etl)$'
    if repo_name and re.match(pattern, repo_name):
        return True, f"Repository name '{repo_name}' is valid."
    return False, f"Repo name '{repo_name}' does not match the required pattern."


def check_is_private_repo(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates if the repository is private."""
    if context.get("bitbucket_repo_is_private", False):
        return True, "The repository is private."
    return False, "The repository must be private to comply with policies."


RULES = [
    {"name": "Source Branch Type", "category": "PR Configuration", "critically": MEDIUM_SEVERITY, "func": check_source_branch_type},
    {"name": "Source Branch Name", "category": "PR Configuration", "critically": MEDIUM_SEVERITY, "func": check_source_branch_name},
    {"name": "Destination Branch Type", "category": "PR Configuration", "critically": MEDIUM_SEVERITY, "func": check_destination_branch_type},
    {"name": "Destination Branch Name", "category": "PR Configuration", "critically": MEDIUM_SEVERITY, "func": check_destination_branch_name},
    {"name": "Commit Format (Conventional)", "category": "Commits", "critically": MEDIUM_SEVERITY, "func": check_conventional_commits},
    {"name": "Changelog Entry (README)", "category": "Files", "critically": LOW_SEVERITY, "func": check_readme_changelog_entry},
    {"name": "Version Consistency", "category": "Files", "critically": CRITICAL_SEVERITY, "func": check_version_consistency},
    {"name": "Repository Name", "category": "Repository Configuration", "critically": MEDIUM_SEVERITY, "func": check_repo_name},
    {"name": "Repository Privacy", "category": "Repository Configuration", "critically": CRITICAL_SEVERITY, "func": check_is_private_repo},
]


def main():
    """
    Main function that executes the PR rule validation.
    """
    log_message("PULL REQUEST RULES REPORT", "HEADER")
    pr_context = get_context()

    results = []
    total_failures = 0
    critical_failures = 0

    try:
        for rule in RULES:
            is_valid, message = rule["func"](pr_context)
            status = "PASS" if is_valid else "FAIL"
            if not is_valid:
                total_failures += 1
                if rule["critically"] == CRITICAL_SEVERITY:
                    critical_failures += 1
            results.append({
                "group": rule["category"],
                "rule": rule["name"],
                "status": status,
                "message": message,
                "critically": rule["critically"]
            })
    except Exception as e:
        log_message(f"Unrecoverable error while executing rules: {e}", "ERROR")
        sys.exit(1)

    # Print report
    current_group = ""
    for res in sorted(results, key=lambda x: x['group']):
        if res['group'] != current_group:
            print(f"\n{Colors.BOLD}>> Group: {res['group']}{Colors.ENDC}")
            current_group = res['group']

        status_color = STATUS_COLORS.get(res['status'], Colors.ENDC)
        full_rule_text = f"{res['rule']:<35} (Critically: {res['critically']})"
        message = f"-> {res['message']}" if res['status'] == 'FAIL' else ""

        print(f"  [{status_color}{res['status']:<4}{Colors.ENDC}] {full_rule_text:<55} {Colors.YELLOW}{message}{Colors.ENDC}")

    # Final summary and exit status
    log_message("END OF REPORT", "HEADER")

    if critical_failures > 0:
        log_message(f"RESULT: FAILED. Found {critical_failures} critical error(s).", "ERROR")
        sys.exit(1)
    elif total_failures > MAX_FAILURES_ALLOWED:
        log_message(f"RESULT: FAILED. Found {total_failures} failures. Exceeded threshold of {MAX_FAILURES_ALLOWED}.", "ERROR")
        sys.exit(1)
    elif total_failures > 0:
        log_message(f"RESULT: PASSED WITH WARNINGS. Found {total_failures} non-critical failure(s).", "WARNING")
    else:
        log_message("RESULT: PASSED. All rules were met.", "SUCCESS")


if __name__ == "__main__":
    main()
