import os
import json
from collections import defaultdict


def analyze_file_tree(
    dataset_root_dir: str, output_dir: str, max_files_to_list: int = 10
):
    """
    Analyzes the file tree structure and saves the result to a specified output directory.

    Args:
        dataset_root_dir: The root directory path of the dataset to analyze.
        output_dir: The directory where the output JSON file will be saved.
        max_files_to_list: The threshold for summarizing files of the same type.
    """

    def _recursive_analyze(current_dir):
        # Use the name of the directory being analyzed as the root name
        dir_info = {
            "name": os.path.basename(current_dir),
            "type": "directory",
            "children": [],
        }
        file_groups = defaultdict(list)
        sub_dirs = []

        try:
            for entry in os.scandir(current_dir):
                if entry.is_dir(follow_symlinks=False):
                    sub_dirs.append(_recursive_analyze(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    try:
                        _, ext = os.path.splitext(entry.name)
                        file_ext = ext.lower() if ext else ".<no_ext>"
                        file_info = {
                            "name": entry.name,
                            "type": "file",
                            "size": entry.stat(follow_symlinks=False).st_size,
                        }
                        file_groups[file_ext].append(file_info)
                    except OSError as stat_error:
                        print(f"警告：无法获取文件信息 '{entry.path}': {stat_error}")
        except OSError as scan_error:
            print(f"错误：无法扫描目录 '{current_dir}': {scan_error}")
            dir_info["error"] = f"Could not scan directory: {scan_error}"
            return dir_info

        processed_children = []
        for ext, files in file_groups.items():
            if len(files) > max_files_to_list:
                processed_children.extend(files[:max_files_to_list])
                summary_node = {
                    "name": f"Summary_{len(files)}_{ext.replace('.', '')}_files",
                    "type": "summary",
                    "description": f"{len(files)} files of type '{ext}'",
                    "count": len(files),
                    "extension": ext,
                }
                processed_children.append(summary_node)
            else:
                processed_children.extend(files)

        dir_info["children"] = sub_dirs + processed_children
        return dir_info

    if not os.path.isdir(dataset_root_dir):
        error_msg = f"Input path is not a valid directory: {dataset_root_dir}"
        print(f"错误：{error_msg}")
        return {"error": error_msg}

    # Start the analysis from the root directory
    tree_structure = _recursive_analyze(dataset_root_dir)
    # The root of the JSON should be the directory we analyzed, not a parent
    final_json_structure = tree_structure

    # Save the output to the directory provided by the agent
    dataset_basename = os.path.basename(dataset_root_dir.rstrip("/\\"))
    output_path = os.path.join(output_dir, f"{dataset_basename}_raw_file_tree.json")
    print(f"保存文件树结构到: {output_path}")

    try:
        # Wrap the result in a 'root' key to match Pydantic model expectations
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"root": final_json_structure}, f, ensure_ascii=False, indent=2)
    except IOError as write_error:
        error_msg = f"Failed to write output file: {write_error}"
        print(f"错误：{error_msg}")
        final_json_structure["write_error"] = error_msg

    # Return the structure that will be passed to the LLM, wrapped in 'root'
    return {"root": final_json_structure}
