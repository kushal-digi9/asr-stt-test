import logging
from typing import List, Dict, Any, Optional
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluate RAG pipeline using RAGAS metrics."""
    
    def __init__(self):
        """Initialize RAG evaluator."""
        logger.info("📊 Initializing RAG Evaluator")
        
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        logger.info(f"📊 Loaded {len(self.metrics)} evaluation metrics")
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response.
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved context documents
            ground_truth: Optional ground truth answer
            
        Returns:
            Dict of metric scores
        """
        logger.info(f"📊 Evaluating single RAG response")
        
        # Prepare data
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        try:
            result = evaluate(dataset, metrics=self.metrics)
            scores = result.to_pandas().to_dict('records')[0]
            
            logger.info(f"📊 Evaluation scores: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {}
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple RAG responses.
        
        Args:
            questions: List of user queries
            answers: List of generated answers
            contexts: List of retrieved contexts for each query
            ground_truths: Optional ground truth answers
            
        Returns:
            Dict with aggregated metrics
        """
        logger.info(f"📊 Evaluating batch of {len(questions)} RAG responses")
        
        # Prepare data
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        try:
            result = evaluate(dataset, metrics=self.metrics)
            df = result.to_pandas()
            
            # Calculate aggregated metrics
            aggregated = {
                "mean_scores": df.mean().to_dict(),
                "median_scores": df.median().to_dict(),
                "std_scores": df.std().to_dict(),
                "individual_scores": df.to_dict('records')
            }
            
            logger.info(f"📊 Batch evaluation completed")
            logger.info(f"📊 Mean scores: {aggregated['mean_scores']}")
            
            return aggregated
            
        except Exception as e:
            logger.error(f"❌ Batch evaluation failed: {e}")
            return {}
    
    def create_evaluation_report(
        self,
        eval_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a formatted evaluation report.
        
        Args:
            eval_results: Evaluation results from evaluate_batch
            output_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "RAG EVALUATION REPORT",
            "=" * 60,
            ""
        ]
        
        if "mean_scores" in eval_results:
            report_lines.append("MEAN SCORES:")
            for metric, score in eval_results["mean_scores"].items():
                report_lines.append(f"  {metric}: {score:.4f}")
            report_lines.append("")
        
        if "median_scores" in eval_results:
            report_lines.append("MEDIAN SCORES:")
            for metric, score in eval_results["median_scores"].items():
                report_lines.append(f"  {metric}: {score:.4f}")
            report_lines.append("")
        
        if "std_scores" in eval_results:
            report_lines.append("STANDARD DEVIATION:")
            for metric, score in eval_results["std_scores"].items():
                report_lines.append(f"  {metric}: {score:.4f}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"📊 Report saved to: {output_path}")
        
        return report
