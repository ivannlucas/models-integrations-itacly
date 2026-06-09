# confluence.py
# Handles all interactions with the Confluence API.

import os
from typing import Optional, List

import markdown2
from atlassian import Confluence

from ui import log_message  # Assuming ui.py is in the same project path

# --- Configuration from Environment Variables ---
# Load credentials securely from the environment.
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")


def get_confluence_client() -> Optional[Confluence]:
    """
    Initializes and returns a Confluence API client if credentials are set
    in the environment variables.
    """
    if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
        log_message("One or more Confluence environment variables are missing.", "ERROR")
        return None

    try:
        confluence = Confluence(
            url=CONFLUENCE_URL,
            username=CONFLUENCE_USERNAME,
            password=CONFLUENCE_API_TOKEN,
            cloud=True
        )
        confluence.get_all_spaces(limit=1)
        log_message("Successfully connected to Confluence.", "SUCCESS")
        return confluence
    except Exception as e:
        log_message(f"Failed to initialize or connect with Confluence client: {e}", "ERROR")
        return None


def ensure_page_hierarchy_exists(client: Confluence, space_key: str, hierarchy: List[str],
                                 base_parent_id: Optional[str] = None) -> Optional[str]:
    """
    Ensures a hierarchy of pages exists in Confluence and returns the ID of the last page.
    For a hierarchy ['folder1', 'folder2'], it ensures 'folder2' exists under 'folder1'.
    """
    current_parent_id = base_parent_id

    for page_title in hierarchy:
        log_message(f"Checking for parent page '{page_title}' under parent ID '{current_parent_id or 'space root'}'...",
                    "INFO")

        query = f'space = "{space_key}" AND title = "{page_title}"'
        if current_parent_id:
            query += f' AND parent = {current_parent_id}'

        search_results = client.cql(query, limit=1).get('results', [])

        if search_results:
            found_page = search_results[0]
            # --- FIX: Access the ID from the nested 'content' object ---
            current_parent_id = found_page['content']['id']
            log_message(f"Page '{page_title}' found with ID {current_parent_id}.", "SUCCESS")
        else:
            log_message(f"Page '{page_title}' not found. Creating it...", "INFO")
            try:
                new_page = client.create_page(
                    space=space_key,
                    title=page_title,
                    body="<p>This is an auto-generated container page for documentation.</p>",
                    parent_id=current_parent_id,
                    representation='storage'
                )
                current_parent_id = new_page['id']
                log_message(f"Successfully created page '{page_title}' with ID {current_parent_id}.", "SUCCESS")
            except Exception as e:
                log_message(f"Failed to create page '{page_title}': {e}", "ERROR")
                return None

    return current_parent_id


def create_or_update_confluence_page(client: Confluence, space_key: str, title: str, body: str,
                                     parent_id: Optional[str] = None):
    """
    Creates a new page in Confluence or updates it if it already exists.
    Converts Markdown body to Confluence Storage Format (XHTML).
    """
    try:
        html_body = markdown2.markdown(body, extras=["fenced-code-blocks", "tables", "cuddled-lists"])

        query = f'space = "{space_key}" AND title = "{title}"'
        if parent_id:
            query += f' AND parent = {parent_id}'

        search_results = client.cql(query, limit=1).get('results', [])
        page = search_results[0] if search_results else None

        if page:
            page_id = page['content']['id']  # Also apply fix here for consistency
            client.update_page(
                page_id=page_id,
                title=title,
                body=html_body,
                parent_id=parent_id,
                representation='storage'
            )
            log_message(f"Successfully updated page '{title}' in space '{space_key}'.", "SUCCESS")
        else:
            client.create_page(
                space=space_key,
                title=title,
                body=html_body,
                parent_id=parent_id,
                representation='storage'
            )
            log_message(f"Successfully created page '{title}' in space '{space_key}'.", "SUCCESS")

    except Exception as e:
        log_message(f"An error occurred while publishing to Confluence: {e}", "ERROR")
        raise
