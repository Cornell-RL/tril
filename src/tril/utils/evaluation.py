from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from tril.base_metric import BaseMetric
from tril.base_task import Sample
from tril.logging import Tracker
from tril.metrics.metric_utils import compute_single_metric


@torch.no_grad()
def evaluate_on_samples(
    agent,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    max_prompt_length: int,
    metrics: List[BaseMetric],
    epoch: int,
    split_name: str,
    accelerator: Accelerator,
    tracker: Tracker = None,
    dt_control_token: str = "",
    gen_kwargs: Dict[str, Any] = None,
    positive_samples=None,
):
    # TODO: Don't hardcode
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    # put in eval mode
    agent.eval()

    # tracker
    tracker.log_info("DISTRIBUTED INFERENCE STARTED")

    # generate text by batch
    generations_by_sample_ids = {}
    for batch in tqdm(
        dataloader, desc="DIST INFERENCE", disable=not accelerator.is_local_main_process
    ):
        batch_sample_ids, batch_generated_texts = generate_text(
            agent,
            tokenizer,
            batch,
            accelerator,
            max_prompt_length,
            dt_control_token,
            gen_kwargs,
        )
        for sample_id, gen_text in zip(batch_sample_ids, batch_generated_texts):
            generations_by_sample_ids[sample_id] = gen_text

    # tracker
    tracker.log_info("DISTRIBUTED INFERENCE FINISHED")

    # compute metrics
    sample_predictions_dict, corpus_level_metrics = compute_metrics(
        dataloader, metrics, generations_by_sample_ids, split_name, agent, accelerator
    )

    if tracker is not None:
        # log the entire predictions
        tracker.log_predictions(epoch, split_name, sample_predictions_dict)
        # log the corpus level scores
        tracker.log_metrics(epoch, split_name, corpus_level_metrics)

    # back to train mode
    agent.train()


def compute_metrics(
    dataloader: DataLoader,
    metrics: List[BaseMetric],
    generations_by_sample_ids: Dict[int, str],
    split_name: str,
    agent,
    accelerator: Accelerator,
    positive_by_sample_ids=None,
):
    # gather everything
    all_ref_texts = []
    all_prompt_texts = []
    all_meta_infos = []
    all_generated_texts = []
    all_sample_ids = []
    for sample in dataloader.dataset:
        all_ref_texts.append(sample.references)
        all_prompt_texts.append(sample.prompt_or_input_text)
        all_generated_texts.append(generations_by_sample_ids[sample.id])
        all_meta_infos.append(sample.meta_data)
        all_sample_ids.append(sample.id)

    # gather metrics
    corpus_level_metrics = {}
    sample_scores_by_metric = {}
    n_samples = len(all_sample_ids)
    if metrics is not None:
        for metric in metrics:
            metric_dict = compute_single_metric(
                metric,
                all_sample_ids,
                all_prompt_texts,
                all_generated_texts,
                all_ref_texts,
                all_meta_infos,
                agent,
                split_name,
                accelerator,
            )

            for metric_key, (sample_scores, corpus_score) in metric_dict.items():
                if sample_scores is None:
                    sample_scores = ["n/a"] * n_samples
                corpus_level_metrics[metric_key] = corpus_score
                sample_scores_by_metric[metric_key] = sample_scores

    # aggregate sample metric scores
    sample_predictions_dict = []
    for ix, (sample_id, prompt_text, generated_text, ref_texts) in enumerate(
        zip(all_sample_ids, all_prompt_texts, all_generated_texts, all_ref_texts)
    ):
        sample_prediction = {
            "split_name": split_name,
            "sample_id": sample_id,
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "ref_text": "".join(
                [
                    f"<START-{ref_ix+1}>" + ref_text + f"<END-{ref_ix+1}>"
                    for ref_ix, ref_text in enumerate(ref_texts)
                ]
            ),
        }
        for metric_key, sample_scores in sample_scores_by_metric.items():
            # if metric_key != "fluency_metrics/perplexity":
            sample_prediction[metric_key] = sample_scores[ix]
        sample_predictions_dict.append(sample_prediction)

    return sample_predictions_dict, corpus_level_metrics


def generate_text(
    agent,
    tokenizer: AutoTokenizer,
    samples: List[Sample],
    accelerator: Accelerator,
    max_prompt_length: int,
    dt_control_token: str,
    gen_kwargs: Dict[str, Any],
):
    prompt_texts = [
        dt_control_token + sample.prompt_or_input_text for sample in samples
    ]
    sample_ids = torch.tensor([sample.id for sample in samples]).to(accelerator.device)
    generated_output = accelerator.unwrap_model(agent.policy).eval_generate(
        tokenizer,
        accelerator,
        prompt_texts,
        sample_ids,
        max_prompt_length,
        gen_kwargs=gen_kwargs,
    )
    return generated_output.sample_ids, generated_output.gen_texts
