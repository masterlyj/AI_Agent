import requests
import os
import time
import glob
import json
from dotenv import load_dotenv
import zipfile
import io
import logging

# 配置 logger
logger = logging.getLogger("mineru")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

class MineruProcessor:
    """
    一个用于处理Mineru PDF解析任务的客户端。

    这个类封装了与Mineru API交互的所有步骤，包括：
    1. 申请上传URL。
    2. 上传PDF文件（自动分批）。
    3. 轮询等待解析结果。
    4. 下载并解压结果，最终保存为Markdown文件。

    用法:
        processor = MineruProcessor(api_key="YOUR_API_KEY")
        processor.process_directory("path/to/dataset", "path/to/output")
    """

    def __init__(self, api_key: str, 
                 base_url: str = "https://mineru.net/api/v4", 
                 batch_size: int = 50, 
                 timeout_seconds: int = 1800, 
                 polling_interval: int = 10):
        """
        初始化Mineru处理器。

        Args:
            api_key (str): 你从Mineru官网申请的API密钥。
            base_url (str, optional): API的基础URL。默认为 "https://mineru.net/api/v4"。
            batch_size (int, optional): 每次批量处理的文件数量。默认为 50。
            timeout_seconds (int, optional): 轮询等待的超时时间（秒）。默认为 1800。
            polling_interval (int, optional): 每次轮询的间隔时间（秒）。默认为 10。
        """
        if not api_key:
            raise ValueError("API apy_key 不能为空。")
            
        self.api_key = api_key
        self.base_url = base_url
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.polling_interval = polling_interval
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _get_pdf_file_paths(self, root_dir: str) -> list:
        logger.info(f"正在从 '{root_dir}' 目录中查找所有 .pdf 文件...")
        file_paths = glob.glob(os.path.join(root_dir, "**", "*.pdf"), recursive=True)
        logger.info(f"成功找到 {len(file_paths)} 个PDF文件。")
        return file_paths

    def _ensure_dir_exists(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _get_upload_urls(self, file_paths: list) -> tuple:
        logger.info("--- 步骤 1: 正在向Mineru API申请文件上传URL ---")
        url = f"{self.base_url}/file-urls/batch"
        files_data = [{"name": os.path.basename(p), "is_ocr": True, "data_id": p} for p in file_paths]
        payload = {"enable_formula": True, "language": "ch", "enable_table": True, "files": files_data, "model_version": "vlm"}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get("code") == 0:
            batch_id = result["data"]["batch_id"]
            file_urls = result["data"]["file_urls"]
            logger.info(f"成功获取上传URL。Batch ID: {batch_id}")
            return batch_id, file_urls
        else:
            raise Exception(f"申请上传URL失败: {result.get('msg', '未知错误')}")

    def _upload_files(self, local_files: list, upload_urls: list):
        logger.info("--- 步骤 2: 正在上传所有PDF文件 ---")
        if len(local_files) != len(upload_urls):
            logger.warning(f"文件列表数量({len(local_files)})与URL列表数量({len(upload_urls)})不匹配，跳过上传。")
            return
        upload_count = 0
        for local_path, upload_url in zip(local_files, upload_urls):
            try:
                with open(local_path, 'rb') as f:
                    upload_response = requests.put(upload_url, data=f)
                    upload_response.raise_for_status()
                    logger.info(f"  - {os.path.basename(local_path)} ... 上传成功")
                    upload_count += 1
            except Exception as e:
                logger.error(f"  - {os.path.basename(local_path)} ... 上传失败! 错误: {e}")
        if upload_count == len(local_files):
            logger.info("当前批次的所有文件均已成功上传。")

    def _poll_for_results(self, batch_id: str) -> list:
        logger.info("--- 步骤 3: 正在等待解析结果 (此过程可能需要较长时间) ---")
        url = f"{self.base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()
        while True:
            if time.time() - start_time > self.timeout_seconds:
                raise TimeoutError("轮询解析结果超时。")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            if result.get("code") == 0:
                data = result.get("data", {})
                top_level_status = data.get("status", "SUCCESS").upper()
                results_list = data.get("extract_result")
                if isinstance(results_list, list):
                    pending_files = [item.get('file_name') for item in results_list if item.get("state") not in ['done']]
                    if not pending_files:
                        logger.info("  - 所有文件的状态均已完成！解析任务成功结束！")
                        return results_list
                    else:
                        done_count = len(results_list) - len(pending_files)
                        first_pending_state = next((item.get("state") for item in results_list if item.get("state") not in ['done']), "N/A")
                        logger.info(f"  - 批次处理中: {done_count}/{len(results_list)} 个文件已完成。仍在等待 {len(pending_files)} 个文件... (例如: {pending_files[0]} 状态为 '{first_pending_state}')")
                else:
                    logger.info(f"  - 批次状态为SUCCESS，但尚未返回文件列表，继续等待...")
            else:
                logger.info(f"  - 任务仍在处理中... (API code: {result.get('code')}, msg: {result.get('msg')})")
            time.sleep(self.polling_interval)

    def _save_results(self, file_results_list: list, output_dir: str, source_dir: str):
        logger.info("--- 步骤 4: 正在下载、解压并保存解析结果 ---")
        self._ensure_dir_exists(output_dir)
        for file_result in file_results_list:
            original_path = file_result.get('data_id')
            zip_url = file_result.get('full_zip_url')
            state = file_result.get('state')
            if not original_path: continue
            base_name = os.path.basename(original_path)
            if state != 'done' or not zip_url:
                logger.warning(f" 文件 {base_name} 处理未成功或没有zip链接，状态: '{state}'，已跳过。")
                continue
            try:
                relative_path = os.path.relpath(original_path, source_dir)
                relative_md_path = os.path.splitext(relative_path)[0] + ".md"
                output_filepath = os.path.join(output_dir, relative_md_path)
                output_subfolder = os.path.dirname(output_filepath)
                self._ensure_dir_exists(output_subfolder)
                logger.info(f"  - 正在下载 {base_name} 的结果包...")
                zip_response = requests.get(
                    zip_url,
                    proxies={"http": None, "https": None},
                    timeout=30
                )
                zip_response.raise_for_status()
                zip_in_memory = io.BytesIO(zip_response.content)
                with zipfile.ZipFile(zip_in_memory, 'r') as zf:
                    md_files_in_zip = [f for f in zf.namelist() if f.lower().endswith('.md')]
                    if not md_files_in_zip: continue
                    markdown_filename = md_files_in_zip[0]
                    logger.info(f"  - 正在从ZIP中提取: {markdown_filename}")
                    markdown_content = zf.read(markdown_filename).decode('utf-8')
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    logger.info(f"  - 已成功保存至: {output_filepath}")
            except Exception as e:
                logger.error(f" 处理 {base_name} 的结果时发生意外: {e}")
        logger.info("当前批次的所有结果均已处理完毕。")

    def process_directory(self, source_dir: str, output_dir: str):
        """
        处理指定目录下的所有PDF文件。

        这是该类的主要入口方法。它会自动完成查找文件、分批、
        上传、轮询和保存结果的全过程。

        Args:
            source_dir (str): 包含PDF文件的源目录路径 (例如 "dataset")。
            output_dir (str): 用于保存Markdown结果的输出目录路径 (例如 "output_markdown")。
        """
        try:
            all_files = self._get_pdf_file_paths(source_dir)
            if not all_files:
                return

            total_files = len(all_files)
            total_batches = (total_files + self.batch_size - 1) // self.batch_size
            logger.info(f"\n文件总数: {total_files}。将分为 {total_batches} 批处理，每批最多 {self.batch_size} 个文件。")

            for i in range(total_batches):
                start_index = i * self.batch_size
                end_index = start_index + self.batch_size
                file_batch = all_files[start_index:end_index]

                logger.info(f"\n{'='*20} 正在处理第 {i+1}/{total_batches} 批文件 ({len(file_batch)}个) {'='*20}")

                batch_id, file_urls = self._get_upload_urls(file_batch)
                self._upload_files(file_batch, file_urls)
                final_results = self._poll_for_results(batch_id)
                self._save_results(final_results, output_dir, source_dir)

            logger.info(f"\n🎉 全部 {total_batches} 批任务均已成功完成！")

        except requests.exceptions.HTTPError as err:
            logger.error(f"\n[错误] HTTP请求失败: {err.response.status_code} {err.response.text}")
        except requests.exceptions.RequestException as err:
            logger.error(f"\n[错误] 网络连接失败: {err}")
        except Exception as err:
            logger.error(f"\n[错误] 程序执行出错: {err}")

if __name__ == "__main__":
    load_dotenv()
    
    MINERU_API_KEY = os.getenv("MINERU_API_KEY")

    if not MINERU_API_KEY:
        logger.error("错误：请确保 .env 文件中已设置 MINERU_API_KEY。")
    else:
        # 1. 定义输入和输出目录
        pdf_source_directory = r"data\inputs"
        markdown_output_directory = r"data\outputs"

        # 2. 创建 MineruProcessor 实例
        # 可以在这里自定义参数，例如 processor = MineruProcessor(api_key=MINERU_API_KEY, batch_size=20)
        processor = MineruProcessor(api_key=MINERU_API_KEY)

        # 3. 调用主方法，启动处理流程
        processor.process_directory(source_dir=pdf_source_directory, output_dir=markdown_output_directory)