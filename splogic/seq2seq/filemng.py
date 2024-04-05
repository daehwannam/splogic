
"filemng: File Management"

import os
import tempfile

import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

from dhnamlib.pylib import filesys
from dhnamlib.pylib.decoration import curry
from dhnamlib.pylib.mllib.learning import get_measure, CheckpointManager
from dhnamlib.pylib.context import must_skipped, skip_if_possible

from splogic.utility.acceleration import accelerator


skip_if_not_wlmp = skip_if_possible  # wlmp == within local main process
replace_dir = accelerator.within_local_main_process(curry(filesys.replace_dir)(strict=False), must_skipped)
prepare_dir = accelerator.within_local_main_process(filesys.prepare_dir, must_skipped)
copy_dir = accelerator.within_local_main_process(curry(filesys.copy_dir)(replacing=True, deep=False))
mkloc_unless_exist = accelerator.within_local_main_process(filesys.mkloc_unless_exist)
make_symlink = accelerator.within_local_main_process(filesys.make_symlink)
change_symlink = accelerator.within_local_main_process(curry(filesys.change_symlink)(strict=False))
copy_symlink = accelerator.within_local_main_process(curry(filesys.copy_symlink)(replacing=True))
mkdtemp = accelerator.within_local_main_process(tempfile.mkdtemp)
rename_dir = accelerator.within_local_main_process(os.rename)


class AcceleratedCheckpointManager(CheckpointManager):
    @accelerator.within_local_main_process
    def get_new_checkpoint_path(self):
        return super().get_new_checkpoint_path()

    @accelerator.within_local_main_process
    def clean(self):
        super().clean()


def is_finetuned(pretrained_model_name_or_path):
    return os.path.isfile(os.path.join(pretrained_model_name_or_path, '.finetuned'))


def load_tokenizer(pretrained_model_name_or_path, add_prefix_space, non_nl_tokens=None, sorting_non_nl_tokens=True):
    tokenizer = BartTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        add_prefix_space=add_prefix_space)
    if non_nl_tokens is not None:
        if is_finetuned(pretrained_model_name_or_path):
            assert all(token_id is not None for token_id in tokenizer.convert_tokens_to_ids(non_nl_tokens))
        else:
            # Example of addint new tokens:
            # - https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512
            # - https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087
            # - https://github.com/huggingface/transformers/issues/3446#issuecomment-643171894

            ordered_non_nl_tokens = (sorted if sorting_non_nl_tokens else list)(non_nl_tokens)
            tokenizer.add_tokens(ordered_non_nl_tokens)

            # tokenizer.add_tokens(ordered_non_nl_tokens, special_tokens=True)
            # tokenizer.add_special_tokens(dict(additional_special_tokens=ordered_non_nl_tokens))  # https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087

    return tokenizer


def load_model(pretrained_model_name_or_path, num_tokens=None):
    model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    if num_tokens is None:
        assert is_finetuned(pretrained_model_name_or_path)
    else:
        if is_finetuned(pretrained_model_name_or_path):
            assert model.config.vocab_size == num_tokens
        else:
            model.resize_token_embeddings(num_tokens)

    # Not to make `NoRepeatNGramLogitsProcessor` in `GenerationMixin._get_logits_processor`
    model.config.no_repeat_ngram_size = None

    # Disable length_penalty for `BeamSearchScorer`
    model.config.length_penalty = 0

    # Disable early_stopping for `BeamSearchScorer`.
    # - `beam_search` with `early_stopping=True` results in worse performance than `greedy_search`.
    # - `early_stopping` stops `beam_search` when it find `self.num_beams` number of complete sequences regardless of their scores.
    #   Check `from transformers.generation_beam_search.BeamHypotheses.is_done`.
    model.config.early_stopping = False

    # 0 is the default value for MinLengthLogitsProcessor
    model.config.min_length = 0  

    return model


def load_model_config(pretrained_model_name_or_path):
    return BartConfig.from_pretrained(pretrained_model_name_or_path)


@accelerator.within_local_main_process
def save_model(model, dir_path):
    filesys.touch_with_mkpdirs(os.path.join(dir_path, '.finetuned'))
    # model.save_pretrained(dir_path)
    accelerator.save_pretrained_model(model, dir_path)


OPTIMIZER_FILE_NAME = 'optimizer.pt'
SCHEDULER_FILE_NAME = 'scheduler.pt'


def get_optimizer_file_path(dir_path):
    return os.path.join(dir_path, OPTIMIZER_FILE_NAME)


@accelerator.within_local_main_process
def save_optimizer(optimizer, dir_path):
    torch.save(optimizer.state_dict(), get_optimizer_file_path(dir_path))


def load_and_update_optimizer(optimizer, dir_path):
    optimizer.load_state_dict(torch.load(get_optimizer_file_path(dir_path)))


def get_scheduler_file_path(dir_path):
    return os.path.join(dir_path, SCHEDULER_FILE_NAME)


@accelerator.within_local_main_process
def save_scheduler(scheduler, dir_path):
    torch.save(scheduler.state_dict(), get_scheduler_file_path(dir_path))


def load_and_update_scheduler(scheduler, dir_path):
    scheduler.load_state_dict(torch.load(get_scheduler_file_path(dir_path)))


optim_measures = [get_measure('accuracy', True), get_measure('accuracy_fraction', True)]
search_measures = [get_measure('oracle_accuracy', True), get_measure('oracle_accuracy_fraction', True)]


STATUS_FILE_NAME = 'status.json'


@accelerator.within_local_main_process
def save_status(status, dir_path, file_name=STATUS_FILE_NAME):
    filesys.extended_json_pretty_save(status, os.path.join(dir_path, file_name))


NO_DEFAULT = object()


def load_status(dir_path, file_name=STATUS_FILE_NAME, default=NO_DEFAULT):
    file_path = os.path.join(dir_path, file_name)
    if os.path.isfile(file_path):
        return filesys.extended_json_load(os.path.join(dir_path, file_name))
    elif default is NO_DEFAULT:
        raise Exception(f'{file_path} does not exist')
    else:
        return default


@accelerator.within_local_main_process
def save_performance(performance, dir_path, file_name='performance.json'):
    filesys.extended_json_pretty_save(performance, os.path.join(dir_path, file_name))

    updated_performance = dict(performance)
    if 'accuracy' in performance:
        updated_performance.update(accuracy_percent='{:5.2f}'.format(performance['accuracy'] * 100))
    if 'oracle_accuracy' in performance:
        updated_performance.update(oracle_accuracy_percent='{:5.2f}'.format(performance['oracle_accuracy'] * 100))

    name, extension = os.path.splitext(file_name)
    new_file_name = f'{name}-visual{extension}'
    filesys.extended_json_pretty_save(updated_performance, os.path.join(dir_path, new_file_name))


def load_performance(dir_path, file_name='performance.json'):
    return filesys.extended_json_load(os.path.join(dir_path, file_name))


@accelerator.within_local_main_process
def save_analysis(analysis, dir_path, file_name='analysis.json'):
    analysis_file_path = os.path.join(dir_path, file_name)
    filesys.json_pretty_save(analysis, analysis_file_path)


@accelerator.within_local_main_process
def save_predictions(predictions, dir_path):
    predictions_file_path = os.path.join(dir_path, 'predictions.txt')
    filesys.write_lines(predictions_file_path, tuple(map(str, predictions)))


@accelerator.within_local_main_process
def save_time_info(time_info, dir_path, file_name='time_info.json'):
    time_info_file_path = os.path.join(dir_path, file_name)
    filesys.json_pretty_save(time_info, time_info_file_path)


WEAKSUP_FILE_NAME = 'weaksup.jsonl'


@accelerator.within_local_main_process
def save_weaksup_dataset(weaksup_dataset, dir_path, file_name=WEAKSUP_FILE_NAME):
    weaksup_dataset_file_path = os.path.join(dir_path, file_name)
    filesys.jsonl_save(weaksup_dataset, weaksup_dataset_file_path)


def load_weaksup_dataset(dir_path, file_name=WEAKSUP_FILE_NAME):
    weaksup_dataset_file_path = os.path.join(dir_path, file_name)
    return filesys.jsonl_load(weaksup_dataset_file_path)
