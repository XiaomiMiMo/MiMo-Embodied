def eval_seedbench_2_sample_score(sample):
    if sample["pred"].startswith(sample["answer"]):
        return 1
    else:
        return 0


