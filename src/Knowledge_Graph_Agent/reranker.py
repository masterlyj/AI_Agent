import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from BCEmbedding.utils import logger_wrapper
import os
from pathlib import Path
import requests
import json

logger = logger_wrapper('BCEmbedding.models.RerankerModel')

class RerankerModel:
    def __init__(
            self,
            model_name_or_path: str = 'maidalun1020/bce-reranker-base_v1',
            use_fp16: bool = False,
            device: str = None,
            top_k: int = 20,
            **kwargs
    ):
        # 检查是否是本地路径
        is_local_path = os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path)
        
        if is_local_path:
            logger.info(f"正在从本地路径加载模型: {model_name_or_path}")
            # 确保路径存在
            model_path = Path(model_name_or_path)
            if not model_path.exists():
                raise ValueError(f"本地模型路径不存在: {model_name_or_path}")
        else:
            logger.info(f"正在从Hugging Face Hub加载模型: {model_name_or_path}")
        
        # 加载tokenizer和模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                local_files_only=is_local_path,
                **kwargs
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                local_files_only=is_local_path,
                **kwargs
            )
            logger.info(f"✅ 成功加载模型: {model_name_or_path}")
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {e}")
            raise
        
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device

        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16}")

        self.max_length = kwargs.get('max_length', 512)
        self.overlap_tokens = kwargs.get('overlap_tokens', 80)
        self.rerank_top_k = top_k

    def compute_score(
            self,
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
            batch_size: int = 256,
            max_length: int = 512,
            enable_tqdm: bool = True,
            **kwargs
    ):
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        with torch.no_grad():
            scores_collection = []
            for sentence_id in tqdm(range(0, len(sentence_pairs), batch_size), desc='Calculate scores',
                                    disable=not enable_tqdm):
                sentence_pairs_batch = sentence_pairs[sentence_id:sentence_id + batch_size]
                inputs = self.tokenizer(
                    sentence_pairs_batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.model(**inputs_on_device, return_dict=True).logits.view(-1, ).float()
                scores = torch.sigmoid(scores)
                scores_collection.extend(scores.cpu().numpy().tolist())

        # 始终返回列表，即使只有一个元素
        return scores_collection

    def rerank(
            self,
            query: str,
            passages: List[str],
            batch_size: int = 256,
            **kwargs
    ):
        """
        对给定的查询和段落列表进行重新排序。
        """
        # 过滤掉无效的段落
        passages = [p for p in passages if isinstance(p, str) and len(p) > 0]
        if not query or not passages:
            return {'rerank_passages': [], 'rerank_scores': [], 'rerank_ids': []}

        # 1. 创建查询和段落的配对
        sentence_pairs = [[query, passage] for passage in passages]

        # 2. 使用 compute_score 方法直接计算所有配对的分数
        all_scores = self.compute_score(sentence_pairs, batch_size=batch_size, **kwargs)
        
        # 确保 all_scores 是列表
        if not isinstance(all_scores, list):
            all_scores = [all_scores]

        # 3. 根据分数进行排序
        # np.argsort 返回的是排序后的原始索引
        sorted_indices = np.argsort(all_scores)[::-1].tolist()

        # 4. 根据排序后的索引重新组织段落和分数
        sorted_passages = [passages[i] for i in sorted_indices]
        sorted_scores = [all_scores[i] for i in sorted_indices]

        return {
            'rerank_passages': sorted_passages,
            'rerank_scores': sorted_scores,
            'rerank_ids': sorted_indices
        }


class VLLMRerankerModel:
    """基于vLLM API的Rerank模型，支持Qwen3-Reranker instruction格式"""
    
    def __init__(
            self,
            base_url: str,
            model: str,
            api_key: str = "EMPTY",
            top_k: int = 20,
            timeout: int = 60,
            instruction: str = "给定一个查询，检索能回答该查询的相关文档",
            **kwargs
    ):
        """
        初始化vLLM Reranker
        
        Args:
            base_url: vLLM服务的base URL，例如 "http://localhost:18890/v1"
            model: 模型名称
            api_key: API密钥（可选，默认为"EMPTY"）
            top_k: 返回的top-k结果数量
            timeout: 请求超时时间（秒）
            instruction: Rerank指令，用于Qwen3-Reranker等模型
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.rerank_top_k = top_k
        self.timeout = timeout
        self.instruction = instruction
        
        # 检查是否为Qwen3-Reranker模型
        self.is_qwen3_reranker = "qwen3-reranker" in model.lower()
        
        logger.info(f"✅ 初始化vLLM Reranker: {model} (base_url={base_url})")
        if self.is_qwen3_reranker:
            logger.info(f"📝 使用Qwen3-Reranker指令格式: {instruction}")
    
    def compute_score(
            self,
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
            **kwargs
    ) -> List[float]:
        """
        计算句子对的相关性分数
        
        Args:
            sentence_pairs: 句子对列表，每个元素为 [query, passage]
        
        Returns:
            分数列表，与输入文档顺序严格对应
        """
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        num_docs = len(sentence_pairs)
        logger.info(f"📊 vLLM Rerank: 一次性处理 {num_docs} 个文档")
        
        try:
            # 调用vLLM rerank API
            url = f"{self.base_url}/rerank"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 提取query和documents
            query = sentence_pairs[0][0] if len(sentence_pairs) > 0 else ""
            documents = [pair[1] for pair in sentence_pairs]
            
            # 为Qwen3-Reranker添加instruction和Document前缀
            if self.is_qwen3_reranker:
                query_with_instruction = f"<Instruct>: {self.instruction}\n<Query>: {query}"
                documents_with_prefix = [f"<Document>: {doc}" for doc in documents]
            else:
                query_with_instruction = query
                documents_with_prefix = documents
            
            payload = {
                "model": self.model,
                "query": query_with_instruction,
                "documents": documents_with_prefix
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                results_list = result.get("results", [])
                
                # 重要：vLLM返回的results是按score排序的，需要按index重新排序
                # 以确保分数列表与输入documents列表顺序一致
                sorted_results = sorted(results_list, key=lambda x: x["index"])
                scores = [item["relevance_score"] for item in sorted_results]
                
                # 验证返回的分数数量与输入文档数量一致
                if len(scores) != num_docs:
                    logger.error(f"❌ API返回的结果数量 ({len(scores)}) 与输入文档数量 ({num_docs}) 不匹配")
                    return [0.0] * num_docs
                
                logger.info(f"✅ vLLM Rerank 完成: {num_docs} 个文档")
                return scores
            else:
                logger.error(f"❌ vLLM Rerank API错误: {response.status_code} - {response.text}")
                return [0.0] * num_docs
                
        except requests.Timeout:
            logger.error(f"❌ vLLM Rerank API超时")
            return [0.0] * num_docs
        except Exception as e:
            logger.error(f"❌ 调用vLLM Rerank API失败: {e}")
            return [0.0] * num_docs
    
    def rerank(
            self,
            query: str,
            passages: List[str],
            **kwargs
    ):
        """
        对给定的查询和段落列表进行重新排序
        
        Args:
            query: 查询文本
            passages: 段落列表
        
        Returns:
            包含重排序结果的字典
        """
        # 过滤掉无效的段落
        passages = [p for p in passages if isinstance(p, str) and len(p) > 0]
        if not query or not passages:
            return {'rerank_passages': [], 'rerank_scores': [], 'rerank_ids': []}
        
        # 创建查询和段落的配对
        sentence_pairs = [[query, passage] for passage in passages]
        
        # 计算分数（一次性处理所有文档）
        all_scores = self.compute_score(sentence_pairs, **kwargs)
        
        # 确保 all_scores 是列表
        if not isinstance(all_scores, list):
            all_scores = [all_scores]
        
        # 根据分数进行排序
        sorted_indices = np.argsort(all_scores)[::-1].tolist()
        
        # 根据排序后的索引重新组织段落和分数
        sorted_passages = [passages[i] for i in sorted_indices]
        sorted_scores = [all_scores[i] for i in sorted_indices]
        
        return {
            'rerank_passages': sorted_passages,
            'rerank_scores': sorted_scores,
            'rerank_ids': sorted_indices
        }