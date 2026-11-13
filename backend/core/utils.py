"""
core/utils.py
"""

def guess_model_category(repo_id: str, tags: list[str] = None) -> str:
    """
    Attempts to categorize a Hugging Face model based on its name and tags.
    Defaults to 'chat' if no specific heuristics match.
    """
    search_string = repo_id.lower()
    if tags:
        search_string += " " + " ".join([t.lower() for t in tags])

    # 1. Code / Programming
    code_keywords = ["code", "coder", "starcoder", "phind", "deepseek-coder", "instruct-code"]
    if any(keyword in search_string for keyword in code_keywords):
        return "code"

    # 2. Reasoning / Math
    reasoning_keywords = ["reason", "math", "think", "r1", "r-1", "distill", "cot", "chain-of-thought"]
    if any(keyword in search_string for keyword in reasoning_keywords):
        return "reasoning"

    # 3. Document / Long Context
    # Looking for high context window indicators often found in names
    document_keywords = ["128k", "32k", "longbox", "yarn", "context"]
    if any(keyword in search_string for keyword in document_keywords):
        return "document"

    # 5. Default Fallback
    return "chat"