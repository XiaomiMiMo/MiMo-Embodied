# Expected results directory structure:
# <RESULTS_DIR>/
#   - <MODEL_1>/
#     - xxx_samples_xxx.jsonl
#     - xxx_results_xxx.json
#     - ...
#   - <MODEL_2>/
#     - xxx_samples_xxx.jsonl
#     - xxx_results_xxx.json
#     - ...
#   - ...

# For loading benchmark data
export LMMS_EVAL_DATA_HOME=""
# For loading unzipped video data
export HF_HOME=""

streamlit run main.py