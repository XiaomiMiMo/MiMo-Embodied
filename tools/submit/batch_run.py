import os
import subprocess
import argparse

task_list = [
    "where2place_point",
    "roboafford",
    "part_affordance",
    "roborefit",
    "vabench_point_box",
    "egoplan",
    "robovqa",
    "cosmos_reason1_boxed",
    "cvbench_boxed",
    "erqa_boxed",
    "embspatialbench",
    "sat",
    "robospatial",
    "refspatialbench",
    "crpe_relation",
    "metavqa_eval",
    "vsibench_boxed",
    "codalm",
    "drama",
    "drive_action_boxed_detail",
    "lingoqa_boxed",
    "nuscenesqa",
    "omnidrive",
    "nuinstruct",
    "drivelm",
    "maplm",
    "bddx",
    "mme_realworld",
    "idkb"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a HF weights")
    parser.add_argument("--output", required=True, help="Path to save eval results")
    parser.add_argument(
        "--disable_thinking_user",
        action="store_true",
        help="Pass this flag to disable thinking user"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # If disable_thinking_user is set, append 'disable' to the output
    output = args.output
    if args.disable_thinking_user:
        if not output.endswith("_disable"):
            output = f"{output}_disable"

    for task in task_list:
        subprocess.run(
            [
                "bash",
                "mimovl_docs/eval_mimo_vl_args.sh",
                args.input,
                task,
                output,
                "true" if args.disable_thinking_user else "false",
            ],
            check=True,
        )

if __name__ == "__main__":
    main()
