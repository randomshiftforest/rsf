from git import Repo
from tempfile import TemporaryDirectory
from shutil import move, rmtree
from os import mkdir, makedirs

root = "in/nab"

with TemporaryDirectory() as tmpdir:
    rmtree(root, ignore_errors=True)
    Repo.clone_from("https://github.com/numenta/NAB", tmpdir)
    move(f"{tmpdir}/data", f"{root}/data")
    mkdir(f"{root}/labels")
    move(f"{tmpdir}/labels/combined_windows.json",
         f"{root}/labels")
