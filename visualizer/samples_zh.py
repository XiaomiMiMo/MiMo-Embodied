import streamlit as st
import pandas as pd
from collections import defaultdict
import io
import re

from st_utils import (
    init_st_session_state, 
    select_models, deselect_models, clear_models, 
    refresh_cache, 
    set_log_dir, 
    string_to_color, 
    is_dark_color,
    draw_annotations,
    get_all_logs
)
from data_utils import (
    load_result, load_samples, process_sample, process_images, load_dataset, 
    load_dataset_info
)

def show_samples():
    st.set_page_config(page_title="VLM评测-样本可视化", page_icon=":material/dashboard:", layout="wide")
    st.title("VLM评测-样本可视化", anchor="overview")

    SUPPORT_DATASETS, SUPPORT_SCORE_DATASETS, METRICS, TASK_NAME_TO_TASK = load_dataset_info()

    init_st_session_state()

    # 在侧边栏选择模型和数据集
    with st.sidebar:
        if st.button("设置评测结果路径", use_container_width=True):
            set_log_dir()
        
        if st.button("刷新评测结果缓存", use_container_width=True):
            refresh_cache()
    
    if st.session_state.log_dir is None:
        st.write("请先设置评测结果路径")
        return
        
    logs, datasets, models = st.session_state.cache

    with st.sidebar:
        st.write("---")
        dataset_selected = st.selectbox("选择数据集", datasets)
        with st.expander("选择模型"):
            model_search = st.text_input("搜索模型")
            filtered_models = [model for model in models if model_search.lower() in model.lower()]
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("全选", use_container_width=True):
                    select_models(filtered_models)
            with col2:
                if st.button("取消", use_container_width=True):
                    deselect_models(filtered_models)
            with col3:
                if st.button("重置", use_container_width=True):
                    clear_models()
            for model in filtered_models:
                checked = st.checkbox(
                    model, 
                    value=model in st.session_state.selected_models,
                    key=f"model-checkbox-{model}"
                )
                if checked and model not in st.session_state.selected_models:
                    st.session_state.selected_models.add(model)
                elif not checked and model in st.session_state.selected_models:
                    st.session_state.selected_models.remove(model)


    st.write(f"**数据集：{dataset_selected}**")

    if len(st.session_state.selected_models) == 0:
        st.write("请选择至少一个模型")
        return

    # 读取选中数据集的
    _all_results = {}
    _all_metrics = set()
    for model in st.session_state.selected_models:
        _result = load_result(logs, model, dataset_selected)
        if _result is not None:
            sample_count_original, sample_count_effective, results = _result
            _all_results[model] = {
                "sample_count_original": sample_count_original,
                "sample_count_effective": sample_count_effective,
                "results": results
            }
            _all_metrics.update(results.keys())

    if len(_all_results) == 0:
        st.write("没有可用的结果")
        return
    
    st.write("总样本数：", list(_all_results.values())[0]["sample_count_original"])

    result_data = {
        "model": [],
        "sample_count": []
    }

    # 绘制表格展示各模型得分
    for model in _all_results:
        result_data["model"].append(model)
        result_data["sample_count"].append(_all_results[model]["sample_count_effective"])
        for metric in _all_metrics:
            metric_head = metric[0] + (" (+)" if metric[1] else " (-)")
            if metric_head not in result_data:
                result_data[metric_head] = []
            if metric not in _all_results[model]["results"]:
                result_data[metric_head].append(None)
            else:
                result_data[metric_head].append(_all_results[model]["results"][metric])

    st.dataframe(result_data, use_container_width=True)

    # 读取所有选中的模型和数据集的输出结果
    _all_samples = {}
    _all_ids = set()
    for model in st.session_state.selected_models:
        _samples = load_samples(logs, model, dataset_selected)
        if _samples is None:
            continue
        _all_samples[model] = _samples
        _all_ids.update(_samples.keys())
    
    # 展示输出结果示例
    with st.expander("查看样本"):
        st.json(list(list(_all_samples.values())[0].values())[0])

    if dataset_selected not in SUPPORT_DATASETS:
        st.write("当前数据集暂不支持可视化样本")
        return
    
    with st.spinner("加载数据中..."):
        # 读取原始数据集
        if st.session_state.cached_dataset.get(dataset_selected) is None:
            st.session_state.cached_dataset[dataset_selected] = load_dataset(TASK_NAME_TO_TASK[dataset_selected])
        cached_dataset = st.session_state.cached_dataset[dataset_selected]

        processed_samples = {}
        all_tags = set()
        
        for model, _samples in _all_samples.items():
            for doc_id in _samples:
                processed_sample = process_sample(_samples[doc_id], TASK_NAME_TO_TASK[dataset_selected], cached_dataset)
                if doc_id not in processed_samples:
                    processed_samples[doc_id] = {}
                    processed_samples[doc_id]["input"] = processed_sample["input"]
                    processed_samples[doc_id]["answer"] = processed_sample["answer"]
                    gt_annotations = processed_sample.get("gt_annotations", None)
                    tags = processed_sample.get("tags", None)
                    if tags is not None:
                        all_tags.update(tags)
                        processed_samples[doc_id]["tags"] = tags
                    if gt_annotations is not None:
                        processed_samples[doc_id]["gt_annotations"] = gt_annotations

                    processed_samples[doc_id]["responses"] = {}
                score = processed_sample.get("score", None)
                annotations = processed_sample.get("annotations", None)
                processed_samples[doc_id]["responses"][model] = {
                    'response': processed_sample["response"],
                    'score': score,
                    'annotations': annotations
                }

    
    filter_tag_ids = set()
    with st.expander("根据标签筛选"):
        all_tags = sorted(list(all_tags))
        selected_tags = st.multiselect("展示所有包含以下标签的样本", all_tags, default=all_tags)
        tag_cnts = {}
        if len(selected_tags) > 0:
            for doc_id in processed_samples:
                if processed_samples[doc_id]["tags"] is None:
                    continue
                for tag in processed_samples[doc_id]["tags"]:
                    tag_cnts[tag] = tag_cnts.get(tag, 0) + 1
                if all(tag not in processed_samples[doc_id]["tags"] for tag in selected_tags):
                    filter_tag_ids.add(doc_id)
            
            tag_scores = {}
            for model in st.session_state.selected_models:
                tag_scores[model] = {}
                for tag in selected_tags:
                    for doc_id, processed_sample in processed_samples.items():
                        if tag in processed_sample["tags"] and model in processed_sample["responses"] and processed_sample["responses"][model]["score"] is not None:
                            if tag not in tag_scores[model]:
                                tag_scores[model][tag] = {}
                            tag_scores[model][tag][doc_id] = processed_sample["responses"][model]["score"]
                            if "average" not in tag_scores[model]:
                                tag_scores[model]["average"] = {}
                            tag_scores[model]["average"][doc_id] = processed_sample["responses"][model]["score"]
            
            for model in tag_scores:
                for tag in tag_scores[model]:
                    if len(tag_scores[model][tag]) > 0:
                        tag_scores[model][tag] = sum(tag_scores[model][tag].values()) / len(tag_scores[model][tag])
                    else:
                        tag_scores[model][tag] = None
            
            for model in st.session_state.selected_models:
                if all(tag_scores[model][tag] is None for tag in tag_scores[model]) or model not in tag_scores:
                    tag_scores.pop(model)
            
            tag_scores["count"] = {}
            for tag in tag_cnts:
                tag_scores["count"][tag] = tag_cnts[tag]
            
            tag_scores_dataframe = pd.DataFrame(tag_scores).T
            _columns = sorted(list(tag_scores_dataframe.columns))
            if "average" in _columns:
                _columns.remove("average")
                _columns.append("average")
            tag_scores_dataframe = tag_scores_dataframe[_columns]
            
            st.dataframe(tag_scores_dataframe, use_container_width=True)

    
    # 设置各模型输出答案阈值
    filter_score_ids = set()
    if dataset_selected in SUPPORT_SCORE_DATASETS:
        thresholds = {}
        with st.expander("根据模型得分筛选"):
            for model in _all_results:
                thresholds[model] = st.slider(f"{model} 得分范围", min_value=0.0, max_value=1.0, value=(0.0, 1.0))
            
            for doc_id in processed_samples:
                responses = processed_samples[doc_id]["responses"]
                for model in responses:
                    if responses[model]["score"] is not None and (responses[model]["score"] < thresholds[model][0] or responses[model]["score"] > thresholds[model][1]):
                        filter_score_ids.add(doc_id)
    else:
        st.write("当前数据集不支持根据模型得分筛选")

    st.write("---")

    filtered_ids = filter_tag_ids | filter_score_ids
    all_ids = sorted(list(_all_ids - filtered_ids))

    if len(all_ids) == 0:
        st.write("没有可展示的样本")
        return

    # 分页展示，每页展示20个样例；在侧边栏显示并更改页码
    total_samples = len(all_ids)
    samples_per_page = 20
    total_pages = (total_samples + samples_per_page - 1) // samples_per_page
    with st.sidebar:
        page_idx = st.number_input("选择页码", min_value=1, max_value=total_pages, value=1, step=1)
        st.write(f"当前页码：{page_idx} / {total_pages}")
    

    def show_data(data):
        all_annotations = []
        if data.get("gt_annotations", None) is not None:
            for anno in data["gt_annotations"]:
                all_annotations.append((anno, "GT", "lightgreen"))

        def make_color_text(text, c=None, bg=None):
            lines = text.split("\n")
            result = []
            for line in lines:
                if line.strip() == "":
                    continue
                # line = f"**{line}**"
                if c is not None:
                    line = f":{c}[{line}]"
                if bg is not None:
                    line = f":{bg}-background[{line}]"
                result.append(line)
            return "\n".join(result)
        st.markdown(f":blue-background[**Input**] " + make_color_text(data['input'], bg="gray"))

        try:
            answer_list = eval(data['answer'])
            if isinstance(answer_list, list):
                answer_list = [make_color_text(answer, bg="gray") for answer in answer_list]
                answer = " / ".join(answer_list)
            else:
                answer = make_color_text(answer_list, bg="gray")
        except:
            answer = make_color_text(data['answer'], bg="gray")
        st.markdown(f":green-background[**Answer**] {answer}")
        for model in data['responses']:
            score = data['responses'][model]["score"]
            response = data['responses'][model]["response"]
            annotations = data['responses'][model]["annotations"]

            if score is None:
                boxed_matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', response[0])
                if len(boxed_matches) == 0:
                    response = make_color_text(response[0], bg="gray")
                else:
                    final_boxed = f"\\boxed{{{boxed_matches[-1]}}}"
                    final_boxed_pos = response[0].rfind(final_boxed)
                    pre_boxed = response[0][:final_boxed_pos]
                    if len(pre_boxed) > 0:
                        pre_boxed = make_color_text(pre_boxed, bg="gray")
                    post_boxed = response[0][final_boxed_pos + len(final_boxed):]
                    if len(post_boxed) > 0:
                        post_boxed = make_color_text(post_boxed, bg="gray")
                    response = pre_boxed + make_color_text(final_boxed, bg="orange") + post_boxed
                st.markdown(f":violet-background[**{model}**] {response}")

                if annotations is not None:
                    for annotation in annotations:
                        all_annotations.append((annotation, model, 'orange'))
            else:
                score = round(score, 2)
                score_color = "red" if score < 0.5 else "green"
                boxed_matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', response[0])
                if len(boxed_matches) == 0:
                    response = make_color_text(response[0], bg="gray", c=score_color)
                else:
                    final_boxed = f"\\boxed{{{boxed_matches[-1]}}}"
                    final_boxed_pos = response[0].rfind(final_boxed)
                    pre_boxed = response[0][:final_boxed_pos]
                    if len(pre_boxed) > 0:
                        pre_boxed = make_color_text(pre_boxed, bg="gray", c=score_color)
                    post_boxed = response[0][final_boxed_pos + len(final_boxed):]
                    if len(post_boxed) > 0:
                        post_boxed = make_color_text(post_boxed, bg="gray", c=score_color)
                    response = pre_boxed + make_color_text(final_boxed, bg="orange", c=score_color) + post_boxed
                st.markdown(f":violet-background[**{model} :{score_color}[**({score})**]**] {response}")

                if annotations is not None:
                    for annotation in annotations:
                        all_annotations.append((annotation, model, 'lightgreen' if score > 0.5 else 'lightcoral'))

        return all_annotations
            
    nav = f"[Overview](#overview)"    

    # 处理并展示输出样本
    page_sample_ids = all_ids[(page_idx - 1) * samples_per_page: page_idx * samples_per_page]
    cnt = (page_idx - 1) * samples_per_page
    for doc_id in page_sample_ids:
        data = {}
        images = None
        # 汇总各模型在当前样本上的输出
        for model, _samples in _all_samples.items():
            if doc_id not in _samples:
                continue
            data = processed_samples[doc_id]
            images = process_images(_samples[doc_id], TASK_NAME_TO_TASK[dataset_selected], cached_dataset)
        
        # 展示样本数据
        st.subheader(f"Sample {cnt+1}", anchor=f"sample-{cnt+1}")

        # 显示标签
        if data.get("tags", None) is not None and len(data["tags"]) > 0:
            tag_html = ""
            for tag in data["tags"]:
                bg_color = string_to_color(tag)
                text_color = "white" if is_dark_color(bg_color) else "black"
                tag_html += f'<span style="display:inline-block; margin-left:8px; padding:3px 10px; border-radius:15px; background-color:{bg_color}; color:{text_color}; font-size:0.8em;">{tag}</span>'
            st.markdown(f"**Tags:**{tag_html}", unsafe_allow_html=True)

        top = f"[Top](#sample-{(page_idx - 1) * samples_per_page + 1})" if cnt - (page_idx - 1) * samples_per_page > 0 else "Top"
        prev = f"[Previous](#sample-{cnt})" if cnt - (page_idx - 1) * samples_per_page > 0 else "Previous"
        next = f"[Next](#sample-{cnt+2})" if cnt - (page_idx - 1) * samples_per_page < len(page_sample_ids) - 1 else "Next"
        bottom = f"[Bottom](#sample-{len(page_sample_ids)})" if cnt - (page_idx - 1) * samples_per_page < len(page_sample_ids) - 1 else "Bottom"
        st.markdown(f"{top}  {prev}  {next}  {bottom}")
        nav += f"  [{' '*(8-len(str(cnt+1))) + str(cnt+1)}](#sample-{cnt+1})"
        with st.container(border=True):
            annotations = show_data(data)
        
        if len(annotations) > 0:
            if images is not None and len(images) > 0:
                if len(images) == 1:
                    images = [draw_annotations(images[0], annotations)]
                else:
                    st.warning("当前数据集包含多张图片，无法展示annotations")
            else:
                # streamlit warning
                st.warning("没有可展示的图片，无法展示annotations")
        
        # 展示图片，分栏展示，每栏不超过4张图
        if images is not None and len(images) > 0:
            num_images = len(images)
            if num_images < 4:
                num_image_cols = num_images
            else:
                num_image_cols = 4
            num_image_rows = (num_images + num_image_cols - 1) // num_image_cols
            for i in range(num_image_rows):
                cols = st.columns(num_image_cols)
                for j in range(num_image_cols):
                    idx = i * num_image_cols + j
                    if idx < num_images:
                        if isinstance(images[idx], io.BytesIO):
                            cols[j].video(images[idx])
                        else:
                            cols[j].write(f"Image size: {images[idx].width}x{images[idx].height}")
                            cols[j].image(images[idx])
        
        st.write("---")
        cnt += 1

    with st.sidebar:
        st.divider()
        st.markdown(nav)
    

show_samples()
