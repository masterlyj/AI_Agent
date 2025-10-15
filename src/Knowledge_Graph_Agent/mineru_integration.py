"""
MinerU 集成功能模块
将PDF解析功能集成到RAG系统中，支持上传PDF文件时自动解析并索引
"""

import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# 导入现有的MinerU处理器
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from mineru_analysis import MineruProcessor

from .utils import logger

class MinerUIntegration:
    """
    MinerU PDF解析集成功能类
    负责处理PDF文件的解析和转换为Markdown格式
    """
    
    def __init__(self, api_key: str, output_dir: str = "data/outputs"):
        """
        初始化MinerU集成处理器
        
        Args:
            api_key: MinerU API密钥
            output_dir: 解析后的Markdown文件输出目录
        """
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化MinerU处理器
        self.processor = MineruProcessor(
            api_key=api_key,
            batch_size=10,  # 减少批次大小以提高响应速度
            timeout_seconds=1200,  # 20分钟超时
            polling_interval=5  # 5秒轮询间隔
        )
        
        logger.info(f"✅ MinerU集成处理器已初始化，输出目录: {self.output_dir}")
    
    async def process_pdfs_async(self, pdf_files: List[str]) -> Dict[str, Any]:
        """
        异步处理PDF文件列表
        
        Args:
            pdf_files: PDF文件路径列表
            
        Returns:
            处理结果字典，包含成功和失败的文件信息
        """
        if not pdf_files:
            return {
                "success": False,
                "message": "未提供PDF文件",
                "processed_files": [],
                "failed_files": [],
                "output_files": []
            }
        
        # 创建临时目录用于处理
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_dir = Path(temp_dir) / "input"
            temp_output_dir = Path(temp_dir) / "output"
            temp_input_dir.mkdir(exist_ok=True)
            temp_output_dir.mkdir(exist_ok=True)
            
            # 复制PDF文件到临时输入目录
            valid_pdf_files = []
            for pdf_file in pdf_files:
                if os.path.exists(pdf_file) and pdf_file.lower().endswith('.pdf'):
                    src_path = Path(pdf_file)
                    dst_path = temp_input_dir / src_path.name
                    shutil.copy2(src_path, dst_path)
                    valid_pdf_files.append(str(dst_path))
                else:
                    logger.warning(f"跳过无效文件: {pdf_file}")
            
            if not valid_pdf_files:
                return {
                    "success": False,
                    "message": "未找到有效的PDF文件",
                    "processed_files": [],
                    "failed_files": [],
                    "output_files": []
                }
            
            try:
                # 在后台线程中运行同步的MinerU处理
                logger.info(f"🔄 开始异步处理 {len(valid_pdf_files)} 个PDF文件...")
                
                # 使用asyncio.to_thread在后台线程运行CPU密集型任务
                result = await asyncio.to_thread(
                    self.processor.process_directory,
                    str(temp_input_dir),
                    str(temp_output_dir)
                )
                
                # 收集处理后的Markdown文件
                output_files = []
                if temp_output_dir.exists():
                    for md_file in temp_output_dir.rglob("*.md"):
                        # 将文件复制到最终输出目录
                        final_path = self.output_dir / md_file.name
                        shutil.copy2(md_file, final_path)
                        output_files.append(str(final_path))
                        logger.info(f"✅ 解析完成: {md_file.name} -> {final_path}")
                
                return {
                    "success": True,
                    "message": f"成功处理 {len(output_files)} 个PDF文件",
                    "processed_files": valid_pdf_files,
                    "failed_files": [],
                    "output_files": output_files
                }
                
            except Exception as e:
                logger.error(f"❌ MinerU处理失败: {e}")
                return {
                    "success": False,
                    "message": f"处理失败: {str(e)}",
                    "processed_files": valid_pdf_files,
                    "failed_files": valid_pdf_files,
                    "output_files": []
                }
    
    def is_pdf_file(self, file_path: str) -> bool:
        """检查文件是否为PDF格式"""
        return file_path.lower().endswith('.pdf') and os.path.exists(file_path)
    
    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名"""
        return [".pdf"]


class SmartDocumentIndexer:
    """
    智能文档索引器
    根据文件类型自动选择处理方式：
    - PDF文件 -> MinerU解析 -> Markdown索引
    - Markdown/Text文件 -> 直接索引
    """
    
    def __init__(self, mineru_api_key: Optional[str] = None):
        """
        初始化智能文档索引器
        
        Args:
            mineru_api_key: MinerU API密钥，如果不提供则只能处理文本文件
        """
        self.mineru_api_key = mineru_api_key
        self.mineru_integration = None
        
        if mineru_api_key:
            self.mineru_integration = MinerUIntegration(api_key=mineru_api_key)
            logger.info("✅ 智能文档索引器已初始化，支持PDF解析")
        else:
            logger.warning("⚠️ 未提供MinerU API密钥，只能处理文本格式文件")
    
    async def process_files_for_indexing(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        处理文件列表，返回可直接索引的文件路径
        
        Args:
            file_paths: 原始文件路径列表
            
        Returns:
            处理结果字典
        """
        if not file_paths:
            return {
                "success": False,
                "message": "未提供文件",
                "files_to_index": [],
                "pdf_processed": [],
                "text_files": []
            }
        
        # 分离不同类型的文件
        pdf_files = []
        text_files = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                continue
                
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.pdf':
                pdf_files.append(file_path)
            elif file_ext in ['.md', '.txt']:
                text_files.append(file_path)
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
        
        all_processed_files = []
        pdf_processed_files = []
        
        # 处理PDF文件
        if pdf_files and self.mineru_integration:
            logger.info(f"📄 发现 {len(pdf_files)} 个PDF文件，开始解析...")
            pdf_result = await self.mineru_integration.process_pdfs_async(pdf_files)
            
            if pdf_result["success"]:
                pdf_processed_files = pdf_result["output_files"]
                all_processed_files.extend(pdf_processed_files)
                logger.info(f"✅ PDF解析完成，生成 {len(pdf_processed_files)} 个Markdown文件")
            else:
                logger.error(f"❌ PDF解析失败: {pdf_result['message']}")
        elif pdf_files and not self.mineru_integration:
            logger.warning(f"⚠️ 发现 {len(pdf_files)} 个PDF文件，但未配置MinerU API密钥，跳过处理")
        
        # 添加文本文件
        if text_files:
            logger.info(f"📄 发现 {len(text_files)} 个文本文件")
            all_processed_files.extend(text_files)
        
        return {
            "success": len(all_processed_files) > 0,
            "message": f"准备索引 {len(all_processed_files)} 个文件",
            "files_to_index": all_processed_files,
            "pdf_processed": pdf_processed_files,
            "text_files": text_files
        }
    
    def get_processing_summary(self, result: Dict[str, Any]) -> str:
        """生成处理结果摘要"""
        if not result["success"]:
            return f"❌ 处理失败: {result['message']}"
        
        summary_parts = []
        
        if result.get("pdf_processed"):
            summary_parts.append(f"PDF解析: {len(result['pdf_processed'])} 个")
        
        if result.get("text_files"):
            summary_parts.append(f"文本文件: {len(result['text_files'])} 个")
        
        if result.get("files_to_index"):
            summary_parts.append(f"总计索引: {len(result['files_to_index'])} 个文件")
        
        return " | ".join(summary_parts) if summary_parts else "无文件需要处理"


# 工具函数
def create_mineru_processor_from_env() -> Optional[MinerUIntegration]:
    """
    从环境变量创建MinerU处理器
    
    Returns:
        MinerUIntegration实例，如果未配置API密钥则返回None
    """
    mineru_api_key = os.getenv("MINERU_API_KEY")
    if not mineru_api_key:
        logger.info("未配置MINERU_API_KEY，PDF解析功能不可用")
        return None
    
    try:
        return MinerUIntegration(api_key=mineru_api_key)
    except Exception as e:
        logger.error(f"创建MinerU集成失败: {e}")
        return None