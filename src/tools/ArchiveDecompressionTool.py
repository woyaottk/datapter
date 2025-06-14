import logging
import os
import shutil
import tarfile
import zipfile
import gzip
import patoolib
from typing import List

SUPPORTED_COMPRESSED_EXTENSIONS = [
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".gz",
    ".rar",
    ".7z",
    ".bz2",
]


def is_compressed_file(file_path: str) -> bool:
    """Checks if a file is a supported compressed archive."""
    return any(
        file_path.lower().endswith(ext) for ext in SUPPORTED_COMPRESSED_EXTENSIONS
    )


def find_all_files(directory: str) -> List[str]:
    """Recursively finds all file paths in a given directory."""
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def decompress_archive(file_path: str, output_dir: str):
    """Decompresses a single archive file to a specified directory."""
    file_path_lower = file_path.lower()
    os.makedirs(output_dir, exist_ok=True)
    try:
        if file_path_lower.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
        elif file_path_lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(output_dir)
        elif file_path_lower.endswith(".gz") and not file_path_lower.endswith(
            ".tar.gz"
        ):
            out_filename = os.path.basename(file_path)[:-3]
            out_filepath = os.path.join(output_dir, out_filename)
            with gzip.open(file_path, "rb") as f_in:
                with open(out_filepath, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            patoolib.extract_archive(file_path, outdir=output_dir, verbosity=-1)
        logging.info(f"  - 成功解压 '{os.path.basename(file_path)}' 到 '{output_dir}'")
    except Exception as e:
        logging.error(f"  - 错误: 解压 '{os.path.basename(file_path)}' 失败: {e}")
        pass


def decompress_and_create_replica(source_path: str, target_dir: str) -> str:
    """
    Copies the source to the target directory and recursively decompresses all archives therein.
    This function operates directly in the target_dir without creating subfolders.

    Args:
        source_path: The path to the source file or directory.
        target_dir: The exact directory to copy to and decompress within.

    Returns:
        The path to the processed target directory.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"输入路径不存在: {source_path}")

    try:
        # If source is a directory, copy its *contents*. If a file, copy the file.
        if os.path.isdir(source_path):
            for item in os.listdir(source_path):
                s_item = os.path.join(source_path, item)
                d_item = os.path.join(target_dir, item)
                if os.path.isdir(s_item):
                    shutil.copytree(s_item, d_item, symlinks=False, ignore=None)
                else:
                    shutil.copy2(s_item, d_item)
        else:  # Source is a file
            shutil.copy2(source_path, target_dir)
        logging.info("源内容已复制到目标目录。")

        # Iteratively decompress archives inside the target_dir
        iteration = 1
        while True:
            logging.info(f"\n--- 第 {iteration} 轮扫描和解压 ---")
            all_files_in_target = find_all_files(target_dir)
            compressed_files = [f for f in all_files_in_target if is_compressed_file(f)]

            if not compressed_files:
                print("未发现新的压缩文件，处理完成。")
                break

            logging.info(f"发现 {len(compressed_files)} 个压缩文件需要处理...")
            for archive_path in compressed_files:
                archive_base_name = os.path.basename(archive_path)
                extract_folder_name = (
                    ".".join(archive_base_name.split(".")[:-1])
                    or f"{archive_base_name}_extracted"
                )
                extract_dir = os.path.join(
                    os.path.dirname(archive_path), extract_folder_name
                )

                decompress_archive(archive_path, extract_dir)
                try:
                    os.remove(archive_path)
                    logging.info(f"  - 已删除原始压缩包: '{archive_base_name}'")
                except OSError as e:
                    logging.error(f"  - 警告: 删除压缩包 '{archive_base_name}' 失败: {e}")
            iteration += 1

        logging.info(f"\n解压和复制完成。最终副本根目录: {target_dir}")
        return target_dir

    except Exception as e:
        logging.error(f"处理过程中发生严重错误: {e}")
        raise ValueError(f"处理失败: {e}")
