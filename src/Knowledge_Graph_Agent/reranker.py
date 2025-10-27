import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from BCEmbedding.utils import logger_wrapper
import os
from pathlib import Path
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

        if len(scores_collection) == 1:
            return scores_collection[0]
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