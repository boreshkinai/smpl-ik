try:
    import git
except ImportError:
    print("WARNING: Git installation could not be found")
    def get_git_commit_id():
        return ""
    
    def get_git_diff():
        return ""
else:
    def get_git_commit_id():
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha
        except git.InvalidGitRepositoryError:
            print("WARNING: Not a Git repository")
            return ""
    
    def get_git_diff():
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.git.diff()
        except git.InvalidGitRepositoryError:
            logging.warning("Not a Git repository")
            return ""
