import glob
import os
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "code"))

from general_utils import parse_video_to_propoposal_dists


ABSTRACT_GOAL_LABELS = {
    1: [
        "go to landmark",
        "take object to landmark",
        "help green agent",
        "hinder green agent",
        "get to green agent",
        "get away from green agent",
    ],
    2: [
        "go to landmark",
        "take object to landmark",
        "help red agent",
        "hinder red agent",
        "get to red agent",
        "get away from red agent",
    ],
}

input_dir = "/Users/mmalik16/Downloads/SocialGNN/SIMPLE-new-main/record/test10s_origbeliefs_MMcode_run3/"
output_file = "SIMPLE_proposal_dists_test10s_origbeliefs_MMcode_run3.pkl"

SIMPLE_dist_all = {}

for file_path in glob.glob(os.path.join(input_dir, "*sim*.pik")):
    with open(file_path, "rb") as f:
        record = pickle.load(f)

    video_id = os.path.basename(file_path)[:23]

    SIMPLE_dist_all[video_id] = parse_video_to_propoposal_dists(
        record["dist_all"],
        ABSTRACT_GOAL_LABELS,
    )

with open(output_file, "wb") as f:
    pickle.dump(SIMPLE_dist_all, f)

print(f"Saved {len(SIMPLE_dist_all)} representations to {output_file}")