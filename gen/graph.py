from shutil import unpack_archive
from os import makedirs

root = "in/graph"
makedirs(root, exist_ok=True)

# generated similar to:
# - https://github.com/ToshikiShawn/spotlight_anomaly_detection
unpack_archive("gen/data/darpa.csv.zip", "in/graph")

# adapted from:
# - http://odds.cs.stonybrook.edu/twittersecurity-dataset/
# - http://odds.cs.stonybrook.edu/twitterworldcup2014-dataset/
unpack_archive("gen/data/twitter.zip", "in/graph")
