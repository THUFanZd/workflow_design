from huggingface_hub import list_repo_tree

def list_repo_dir(path_in_repo: str, repo_id: str = "google/gemma-scope-2b-pt-res", repo_type: str = "model"):
    sub_path_list = []
    for item in list_repo_tree(
        repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
    ):
        sub_path_list.append(item.path)
    return sub_path_list
