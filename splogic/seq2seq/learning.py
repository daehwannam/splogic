from itertools import chain

import torch
import torch.nn.functional as F

# from configuration import config, coc

from dhnamlib.pylib import iteration
from dhnamlib.pylib.iteration import iterate
from dhnamlib.pylib.torchlib.dnn import (
    lengths_to_mask, masked_log_softmax, nll_without_reduction, pad_sequence)

from splogic.utility.acceleration import accelerator
from splogic.utility.trie import DenseSpanTrie

from .decoding import token_id_seq_to_action_seq


def get_param_groups(model, learning_rate, weight_decay, non_decayed_names=None):
    if non_decayed_names is None:
        non_decayed_names = ["bias", "layernorm", "layer_norm"]

    model_named_parameters = tuple(model.named_parameters())

    def is_decayed_param_name(param_name):
        return not any(nd_name in param_name for nd_name in non_decayed_names)

    # 'weight_decay' is for AdamW
    # 'lr' values become LambdaLR's base_lrs which are multiplied by lr_lambdas
    param_groups = [
        dict(params=[param for param_name, param in model_named_parameters
                     if is_decayed_param_name(param_name)],
             weight_decay=weight_decay,
             lr=learning_rate),
        dict(params=[param for param_name, param in model_named_parameters
                     if not is_decayed_param_name(param_name)],
             weight_decay=0.0,
             lr=learning_rate)]

    return param_groups


def labels_to_masks(grammar, labels, utterance_token_ids, except_eos=False, dynamic_binding={}):
    '''
    :param grammar:
    :param labels: a tensor of shape (batch_size, seq_len)
    '''
    assert labels.dim() == 2

    allowed_and_ids_pairs_seqs = []
    seq_lengths = []

    for token_id_seq, utterance_token_id_seq in zip(labels.tolist(), utterance_token_ids.tolist()):
        action_seq = token_id_seq_to_action_seq(grammar, token_id_seq)
        # utterance_span_trie = utterance_token_id_seq_to_span_trie(grammar, utterance_token_id_seq)
        with grammar.dynamic_scope.let(**dynamic_binding):
            _allowed_and_ids_pairs_seq = grammar.search_state_cls.action_seq_to_allowed_and_ids_pairs_seq(action_seq)
        allowed_and_ids_pairs_seq = list(chain(
            [(True, [grammar.lf_tokenizer.bos_token_id])],
            _allowed_and_ids_pairs_seq,
            [(True, [grammar.lf_tokenizer.eos_token_id])]
        ))
        allowed_and_ids_pairs_seqs.append(allowed_and_ids_pairs_seq)
        seq_lengths.append(len(allowed_and_ids_pairs_seq))  # including BOS and EOS

        # Note:
        # `len(candidate_action_ids_seq)` could be the length to either the last reduce action or EOS.
        # Actually, EOS token doesn't need to be considered,
        # since the probability of EOS after the last reduce action is always 1 during beam-search
        # due to logits_processor that ignores all actions except EOS.

    padded_candidate_token_ids_seqs = pad_sequence(allowed_and_ids_pairs_seqs, (True, [grammar.lf_tokenizer.pad_token_id]), dim=1)
    softmax_mask = _allowed_and_ids_pairs_seqs_to_softmax_mask(padded_candidate_token_ids_seqs, len(grammar.lf_tokenizer))
    if except_eos:
        _seq_lengths = list(seq_length - 1 for seq_length in seq_lengths)
    else:
        _seq_lengths = seq_lengths
    nll_mask = lengths_to_mask(_seq_lengths, max_length=max(seq_lengths) + int(except_eos))
    # nll_mask and nll tensor have the same size

    return softmax_mask, nll_mask


def _allowed_and_ids_pairs_seqs_to_softmax_mask(allowed_and_ids_pairs_seqs, vocab_size):
    batch_size = len(allowed_and_ids_pairs_seqs)
    assert iteration.all_same(map(len, allowed_and_ids_pairs_seqs))
    seq_len = len(allowed_and_ids_pairs_seqs[0])

    softmax_mask = torch.full([batch_size, seq_len, vocab_size], fill_value=0, dtype=torch.int64)

    for idx_in_batch, allowed_and_ids_pairs_seq in enumerate(allowed_and_ids_pairs_seqs):
        for idx_in_seq, allowed_and_ids_pairs in enumerate(allowed_and_ids_pairs_seq):
            allowed, token_ids = allowed_and_ids_pairs
            if allowed:
                softmax_mask[idx_in_batch, idx_in_seq, token_ids] = 1
            else:
                softmax_mask[idx_in_batch, idx_in_seq, :] = 1
                softmax_mask[idx_in_batch, idx_in_seq, token_ids] = 0

    return softmax_mask


def utterance_token_id_seq_to_span_trie(grammar, utterance_token_id_seq):
    eos_token_id_idx = iteration.index(utterance_token_id_seq, grammar.utterance_tokenizer.eos_token_id, reverse=True)
    assert utterance_token_id_seq[0] == grammar.utterance_tokenizer.bos_token_id
    _utterance_token_id_seq = utterance_token_id_seq[1: eos_token_id_idx]
    # first_utterance_token = grammar.utterance_tokenizer.convert_ids_to_tokens(_utterance_token_id_seq[0])
    # if not first_utterance_token.startswith('Ġ'):
    #     _utterance_token_id_seq[0] = grammar.utterance_tokenizer.convert_tokens_to_ids('Ġ' + first_utterance_token)
    end_of_seq_id = grammar.reduce_action_id
    utterance_span_trie = DenseSpanTrie(_utterance_token_id_seq, end_of_seq_id)
    return utterance_span_trie


def labels_to_nll_mask(grammar, labels, except_eos=False):
    '''
    :param grammar:
    :param labels: a tensor of shape (batch_size, seq_len)
    '''
    assert labels.dim() == 2

    seq_lengths = []

    for token_id_seq in labels.tolist():
        eos_token_id_idx = iteration.index(token_id_seq, grammar.lf_tokenizer.eos_token_id, reverse=True)
        seq_lengths.append(eos_token_id_idx + int(not except_eos))

    nll_mask = lengths_to_mask(seq_lengths, max_length=max(seq_lengths) + int(except_eos))
    # The "+1" of max_length is for EOS, so nll_mask and nll tensor have the same size

    return nll_mask


def compute_nll_loss(logits, labels, softmax_mask=None, nll_mask=None):
    """
    Compute a negative log likelihood loss
    """
    assert logits.dim() == 3    # batch, seq-length, vocab
    assert labels.dim() == 2    # batch, seq-length
    assert logits.size()[:-1] == labels.size()

    # softmax_mask, nll_mask = labels_to_masks(grammar, labels)

    log_probs = masked_log_softmax(logits, mask=softmax_mask, dim=-1)
    nll = nll_without_reduction(log_probs, labels)

    if nll_mask is None:
        masked_nll = nll
    else:
        masked_nll = nll * nll_mask.to(nll.device)

    loss = masked_nll.sum(dim=-1).mean(dim=0)

    return loss


def compute_nlml_loss(logits, labels, nll_mask, group_lengths, averaging):
    """
    Compute a negative log marginal likelihood loss, which is known as the maximum marginal likelihood (MML) loss
    """
    assert logits.dim() == 3    # batch, seq-length, vocab
    assert labels.dim() == 2    # batch, seq-length
    assert logits.size()[:-1] == labels.size()

    log_probs = masked_log_softmax(logits, mask=None, dim=-1)
    nll = nll_without_reduction(log_probs, labels)
    masked_nll = nll * nll_mask.to(nll.device)
    
    example_nll = masked_nll.sum(dim=-1)  # nll for each example
    example_weight = torch.zeros_like(example_nll)
    assert example_weight.dim() == 1
    next_index = 0
    for group_length in group_lengths:
        prev_index = next_index
        next_index = prev_index + group_length
        example_weight[prev_index: next_index] = F.softmax(example_nll[prev_index: next_index], dim=0)

    num_examples = len(group_lengths)
    _loss = (example_weight * example_nll).sum(dim=0)
    loss = _loss / num_examples if averaging else _loss

    return loss


def ss_forward_backward(grammar, model, batch, softmax_masking):
    "Strong-supervision update"

    batched_input = dict(
        input_ids=batch['utterance_token_ids'].to(accelerator.device),
        attention_mask=batch['attention_mask'].to(accelerator.device),
        decoder_input_ids=batch['decoder_input_ids'].to(accelerator.device))
    batched_output = model(**batched_input)

    logits = batched_output['logits']
    labels = batch['labels'].to(accelerator.device)

    if softmax_masking:
        softmax_mask, nll_mask = labels_to_masks(grammar, labels, batch['utterance_token_ids'])
    else:
        softmax_mask = None
        nll_mask = labels_to_nll_mask(grammar, labels)

    loss = compute_nll_loss(
        logits=logits,
        labels=labels,
        softmax_mask=softmax_mask,
        nll_mask=nll_mask,
    )

    accelerator.backward(loss)

    return loss.item()


def ws_forward_backward(grammar, model, batch):
    "Weak-supervision update"

    batch_size = len(batch['example_id'])

    batch_loss = 0

    sub_batch_iter = iterate(batch['ws_sub_batches'])
    for sub_batch in sub_batch_iter:
        with accelerator.accumulate_if(model, accumulating=bool(sub_batch_iter)):
            batched_input = dict(
                input_ids=sub_batch['utterance_token_ids'].to(accelerator.device),
                attention_mask=sub_batch['attention_mask'].to(accelerator.device),
                decoder_input_ids=sub_batch['decoder_input_ids'].to(accelerator.device))
            batched_output = model(**batched_input)

            logits = batched_output['logits']
            labels = sub_batch['labels'].to(accelerator.device)
            nll_mask = labels_to_nll_mask(grammar, labels)
            group_lengths = sub_batch['action_id_seq_group_len']

            loss = compute_nlml_loss(
                logits=logits,
                labels=labels,
                nll_mask=nll_mask,
                group_lengths=group_lengths,
                averaging=False,
            ) / batch_size

            accelerator.backward(loss)
            batch_loss += loss.item()

    return batch_loss


def utterances_to_ids(grammar, utterances):
    encoded_utterances = grammar.utterance_tokenizer(utterances)
    utterance_token_ids = encoded_utterances['input_ids']
    return utterance_token_ids
