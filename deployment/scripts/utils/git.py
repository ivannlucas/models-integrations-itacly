# utils/git.py
# Description: Shared module for Git operations.

import os
import subprocess


def _run_git_command(command):
    """Helper function to run a git command and return its output."""
    try:
        # Executes the command in a shell, captures output, decodes as text, and checks for errors.
        # The output is stripped of whitespace and split into a list of lines.
        return subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip().split('\n')
    except subprocess.CalledProcessError:
        # If the command fails, return an empty list to avoid crashing.
        return []


def get_modified_files(target_branch="develop"):
    """Gets the list of files modified between the current HEAD and a target branch."""
    # This command shows only the names of files that have been changed.
    modified_files_str = _run_git_command(f"git diff origin/{target_branch}..HEAD --name-only")
    return modified_files_str


def get_commit_messages(target_branch="develop"):
    """Gets commit messages between the current HEAD and a target branch."""
    # --pretty=format:%s extracts only the subject line of the commit message.
    commit_messages_str = _run_git_command(f"git log origin/{target_branch}..HEAD --pretty=format:%s")
    return commit_messages_str


def get_bitbucket_pr_context():
    """
    Gets the PR context from Bitbucket CI environment variables.
    Includes branch info, files, commits, and repository details.
    """
    target_branch = os.getenv('BITBUCKET_PR_DESTINATION_BRANCH', 'develop')
    context = {
        'branch_name': os.getenv('BITBUCKET_BRANCH', ''),
        'modified_files': get_modified_files(target_branch),
        'commit_messages': get_commit_messages(target_branch),
        'source_control': 'bitbucket',
        'bitbucket_project_key': os.getenv('BITBUCKET_PROJECT_KEY', ''),
        'bitbucket_repo_name': os.getenv('BITBUCKET_REPO_SLUG', ''),
        'bitbucket_repo_full_name': os.getenv('BITBUCKET_REPO_FULL_NAME', ''),
        'bitbucket_commit': os.getenv('BITBUCKET_COMMIT', ''),
        'bitbucket_pr_destination_commit': os.getenv('BITBUCKET_PR_DESTINATION_COMMIT', ''),
        'bitbucket_repo_is_private': os.getenv('BITBUCKET_REPO_IS_PRIVATE', 'false').lower() == 'true',
        'ci': os.getenv('CI', 'false').lower() == 'true',
        'bitbucket_workspace': os.getenv('BITBUCKET_WORKSPACE', ''),
        'bitbucket_repo_owner': os.getenv('BITBUCKET_REPO_OWNER', ''),
        'bitbucket_git_http_origin': os.getenv('BITBUCKET_GIT_HTTP_ORIGIN', ''),
        'bitbucket_pr_destination_branch': os.getenv('BITBUCKET_PR_DESTINATION_BRANCH', '')
    }
    return context


def get_pr_context(source_control='bitbucket'):
    """
    Generic function to get PR context from a specified source control system.
    Currently only supports Bitbucket.
    """
    if source_control == 'bitbucket':
        context = get_bitbucket_pr_context()
    else:
        raise ValueError(f"Unsupported source control: {source_control}")
    return context
