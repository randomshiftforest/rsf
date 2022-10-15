from git import Repo
from shutil import rmtree
import os

if os.path.exists("autoperiod"):
    rmtree("autoperiod")
Repo.clone_from("https://github.com/akofke/autoperiod", "autoperiod")
