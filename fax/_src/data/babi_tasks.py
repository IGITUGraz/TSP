"""Utilities for downloading and parsing bAbI task data.
Modified from https://github.com/IGITUGraz/H-Mem/blob/main/data/babi_data.py.
"""
import pickle
import re
import shutil
import string
from itertools import chain
from pathlib import Path
from typing import Any, Union

import numpy as np
import requests

from fax.data.utils import hash_dictionary

CHUNK_SIZE = 8192

tasks = {
    1: 'single_supporting_fact',
    2: 'two_supporting_facts',
    3: 'three_supporting_facts',
    4: 'two_arg_relations',
    5: 'three_arg_relations',
    6: 'yes_no_questions',
    7: 'counting',
    8: 'lists_sets',
    9: 'simple_negation',
    10: 'indefinite_knowledge',
    11: 'basic_coreference',
    12: 'conjunction',
    13: 'compound_coreference',
    14: 'time_reasoning',
    15: 'basic_deduction',
    16: 'basic_induction',
    17: 'positional_reasoning',
    18: 'size_reasoning',
    19: 'path_finding',
    20: 'agents_motivations'
}


def download(base_url: str,
             task_file_name: str,
             datasets_path: str,
             output_file: str):
    """Downloads the data set.
    Returns:
      data_dir: string, the data directory.
    """
    pattern = "([\w-]+)(.tar.gz)"
    matches = re.match(pattern, task_file_name)
    extracted_dir = matches[1]
    base_url = base_url
    file_name = task_file_name
    output_parent = Path(datasets_path)
    output_path = output_parent / output_file
    file_path = output_parent / file_name
    url = base_url + file_name
    if output_path.exists():
        print(f"File: {output_path.as_posix()} already present")
        return output_path
    output_parent.mkdir(parents=True, exist_ok=True)
    if output_parent.exists():
        print(f"Downloading {url} ...")
        print('-')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
        shutil.unpack_archive(file_path, output_parent)
        shutil.move(output_parent / extracted_dir, output_path)
    return output_path


def _get_babi_task_files(data_dir: Path, task_id: int, training_set_size: str):
    if not 0 < task_id < 21:
        raise ValueError(f"task_id is {task_id} but must be between 1 and 20 included")
    data_dir = data_dir / "en-valid" if training_set_size == "1k" else data_dir / "en-valid-10k"
    if not data_dir.exists():
        raise FileNotFoundError(f"File: {data_dir.as_posix()} is not found")
    s = f"qa{task_id}"
    # glob and sort by name => (test, train, valid)
    files = sorted(data_dir.glob(f"{s}_*.txt"))
    return files


def load_babi_task(data_dir: Path, task_id: int, training_set_size: str, mode: str):
    """Loads the nth task. There are 20 task in total.
    Arguments:
      data_dir: Path, the data directory.
      task_id: int, the ID of the task (valid values are in `range(1, 21)`).
      training_set_size: string, the size of the training set to load (`1k` or `10k`, default=`1k`).
      mode: str, choose between "default", "only_supporting", "stateful"
        In "default" mode, all the facts presented before the current question in the story are present facts sequence
        In "only_supporting", only the facts useful to answer to the current question are present in the facts sequence
        In "stateful" mode, only the facts presented between the previous question (or the beginning of the story) and
        the current question are present in the facts sequence, fact are not repeated in subsequents stories
    Returns:
      A Python tuple containing the training and testing data for the task.
    """
    files = _get_babi_task_files(data_dir, task_id, training_set_size)
    test_data, test_full_corpus, test_answer_corpus = _get_stories(files[0], mode)
    train_data, train_full_corpus, train_answer_corpus = _get_stories(files[1], mode)
    valid_data, valid_full_corpus, valid_answer_corpus = _get_stories(files[2], mode)
    # x | y = x union y, where x and y are sets
    full_corpus = train_full_corpus | test_full_corpus | valid_full_corpus
    answer_corpus = test_answer_corpus | train_answer_corpus | valid_answer_corpus
    return train_data, valid_data, test_data, full_corpus, answer_corpus


def _get_stories(file_path: Path, mode: str = "default"):
    """Given a file name, read the file, retrieve the stories, and then convert the sentences into a single
    story.
    Eagerly retrieve the corpus set.
    If only_supporting is true, only the sentences that support the answer are kept.
    Arguments:
      f: string, the file name.
      mode: str, (cf. load_task)
    Returns:
      A list of Python tuples containing stories, queries, answers, and corpus set
    """
    full_corpus = set()
    answer_corpus = set()
    with open(file_path, 'r') as f:
        data = []
        story = []
        for line in f:
            line = line.lower()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story.clear()
            # fun fact: split is the past participle of split
            split_line = line.split('\t')
            if len(split_line) > 1:  # Question
                q, a, supporting = split_line
                q = _tokenize(q)
                a = [a]  # Answer is one vocab word even ie it's actually multiple words.
                full_corpus.update(q + a)
                answer_corpus.update(a)
                if mode == "only_supporting":
                    # Only select the related substory.
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                    # add empty fact to match nid with the size of story
                    story.append('')
                else:
                    # Provide all the substories.
                    substory = story.copy()
                    if mode == "stateful":
                        story.clear()

                data.append((substory, q, a))
            else:  # Regular sentence
                sent = _tokenize(line)
                full_corpus.update(sent)
                story.append(sent)

    return data, full_corpus, answer_corpus


def _tokenize(sent: str):
    """Return the tokens of a sentence including punctuation.
    Arguments:
      sent: iterable, containing the sentence.
    Returns:
      A Python list containing the tokens in the sentence.
    Examples:
    ```python
    tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', 'Where', 'is', 'the', 'apple']
    ```
    """
    sent = sent.strip()
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    return [x.strip() for x in re.split(' ', sent)]


def vectorize_data(data, full_corpus_idx, answer_corpus_idx, max_num_sentences, sentence_size, query_size):
    """
    Vectorize stories, queries and answers.
    If a sentence length < `sentence_size`, the sentence will be padded with `0`s. If a story length <
    `max_num_sentences`, the story will be padded with empty sentences. Empty sentences are 1-D arrays of
    length `sentence_size` filled with `0`s. The answer array is returned as a one-hot encoding.
    Arguments:
      data: iterable, containing stories, queries and answers.
      word_idx: dict, mapping words to unique integers.
      max_num_sentences: int, the maximum number of sentences to extract.
      sentence_size: int, the maximum number of words in a sentence.
      query_size: int, the maximum number of words in a query.
    Returns:
      A Python tuple containing vectorized stories, queries, and answers.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        if len(story) > max_num_sentences:
            continue

        ss = []
        for i, sentence in enumerate(story, 1):
            # Pad to sentence_size, i.e., add nil words, and add story.
            ls = max(0, sentence_size - len(sentence))
            ss.append([full_corpus_idx[w] for w in sentence] + [0] * ls)

        # Make the last word of each sentence the time 'word' which corresponds to vector of lookup table.
        # for i in range(len(ss)):
        #     ss[i][-1] = word_idx[f"time{len(ss) - i}"]

        # Pad stories to max_num_sentences (i.e., add empty stories).
        ls = max(0, max_num_sentences - len(ss))
        for _ in range(ls):
            ss.append([0] * sentence_size)

        # Pad queries to query_size (i.e., add nil words).
        lq = max(0, query_size - len(query))
        q = [full_corpus_idx[w] for w in query] + [0] * lq
        if answer_corpus_idx is None:
            y = full_corpus_idx[answer[0]]
        else:
            y = answer_corpus_idx[answer[0]]

        S.append(ss)
        Q.append(q)
        A.append(y)

    return np.array(S, dtype=np.float32), np.array(Q, dtype=np.float32), np.array(A, dtype=np.int64)


def read_numpy_data(babi_parameters: dict, data_output_path: str):
    data_path = Path(data_output_path)
    data_path = data_path / "raw_data"
    data_dir = hash_dictionary(babi_parameters)
    print(data_dir, flush=True)

    current_path = data_path / data_dir
    if not current_path.is_dir():
        print(f"The babi task for parameters {babi_parameters} does not exist in memory")
        print("Try to create babi task for given parameters...")
        babi_to_numpy_data(babi_parameters, data_output_path)
    x = np.load(current_path / "train_x.npz")
    y = np.load(current_path / "train_y.npy")
    train_set = ((x["arr_0"], x["arr_1"].astype(float)), y)
    x = np.load(current_path / "valid_x.npz")
    y = np.load(current_path / "valid_y.npy")
    valid_set = ((x["arr_0"], x["arr_1"].astype(float)), y)
    x = np.load(current_path / "test_x.npz")
    y = np.load(current_path / "test_y.npy")
    test_set = ((x["arr_0"], x["arr_1"].astype(float)), y)
    with open(current_path / "metadata.pickle", "rb") as fd_r:
        metadata = pickle.load(fd_r)
    return train_set, valid_set, test_set, metadata


def babi_to_numpy_data(babi_parameters: dict, data_output_path: str):
    data_path = Path(data_output_path)
    data_path = data_path / "raw_data"
    data_path.mkdir(parents=True, exist_ok=True)
    data_dir = hash_dictionary(babi_parameters)
    current_path = data_path / data_dir
    if not current_path.is_dir():
        train_set, valid_set, test_set, metadata = babi_task(**babi_parameters)
        current_path.mkdir(parents=True, exist_ok=True)
        x, y = train_set
        np.savez(current_path / "train_x", *x)
        np.save(current_path / "train_y", y)
        x, y = valid_set
        np.savez(current_path / "valid_x", *x)
        np.save(current_path / "valid_y", y)
        x, y = test_set
        np.savez(current_path / "test_x", *x)
        np.save(current_path / "test_y", y)
        with open(current_path / "metadata.pickle", "wb") as fd_w:
            pickle.dump(metadata, fd_w, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Do nothing: Task for {str(babi_parameters)} \nis already present at\n{str(current_path.as_posix())}")


def babi_task(task_id: str,
              training_set_type: str,
              mode: str,
              class_space: str,
              max_num_sentences: int,
              hops: int,
              babi_file_name: str = "tasks_1-20_v1-2.tar.gz",
              datasets_path: str = "./datasets",
              output_file: str = "babi_tasks_1-20_v1-2") -> tuple[
                  tuple[np.ndarray, ...],
                  tuple[np.ndarray, ...],
                  tuple[np.ndarray, ...],
                  dict[str, Any]]:
    base_url: str = "http://www.thespermwhale.com/jaseweston/babi/"
    """
    Download (if needed) and parse babi task train, valid and test sets on standard python structure
    :param task_id:
    :param max_num_sentences:
    :param hops:
    :param training_set_size:
    :param mode:
    :return:
    """
    task_id = task_id.split(",")
    task_id = [int(_id) for _id in task_id]
    output_file = download(base_url, babi_file_name, datasets_path, output_file)
    joint_data = []
    joint_train_data = []
    joint_valid_data = []
    joint_test_data = []
    joint_full_corpus = set()
    joint_answer_corpus = set()
    task_labels_train = []
    task_labels_valid = []
    task_labels_test = []
    for _id in task_id:
        train_data, valid_data, test_data, full_corpus, answer_corpus = load_babi_task(output_file, _id,
                                                                                   training_set_type, mode)
        
        joint_data = train_data + valid_data + test_data + joint_data
        task_labels_train += [_id]*len(train_data)
        task_labels_valid += [_id]*len(valid_data)
        task_labels_test += [_id]*len(test_data)
        joint_train_data += train_data
        joint_valid_data += valid_data
        joint_test_data += test_data
        joint_full_corpus |= full_corpus
        joint_answer_corpus |= answer_corpus
         

    # x | y = x union y where x and y are sets
    # vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    # make padding word explicit
    full_corpus_idx = {'<pad>': 0}
    for i, word in enumerate(joint_full_corpus):
        full_corpus_idx[word] = i + 1
    if class_space == "answer_corpus":
        answer_corpus_idx = {'<pad>': 0}
        for i, word in enumerate(joint_answer_corpus):
            answer_corpus_idx[word] = i + 1
    else:
        answer_corpus_idx = None

    max_story_size = max(map(len, (s for s, _, _ in joint_data)))

    max_num_sentences = max_story_size if max_num_sentences == -1 else min(max_num_sentences,
                                                                           max_story_size)

    # Add time words/indexes
    # what is the additional value of string index here ?
    # why not using integer index and use word_idx["time{len(ss) - i}"] in vectorize_data
    # for i in range(max_num_sentences):
    #     word_idx['time{}'.format(i + 1)] = 'time{}'.format(i + 1)

    # for i in range(max_num_sentences):
    #     current_vocab_size = len(word_idx)
    #     word_idx['time{}'.format(i + 1)] = current_vocab_size

    vocab_size = len(joint_full_corpus) + 1
    answer_vocab_size = len(joint_answer_corpus) + 1
    max_sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in joint_data)))  #

    max_query_size = max(map(len, (q for _, q, _ in joint_data)))
    # Vectorize the data.
    max_words = max(max_sentence_size, max_query_size)

    trainS, trainQ, trainA = vectorize_data(
        joint_train_data, full_corpus_idx, answer_corpus_idx,
        max_num_sentences, max_words, max_words)
    validS, validQ, validA = vectorize_data(
        joint_valid_data, full_corpus_idx, answer_corpus_idx,
        max_num_sentences, max_words, max_words)
    testS, testQ, testA = vectorize_data(
        joint_test_data, full_corpus_idx, answer_corpus_idx,
        max_num_sentences, max_words, max_words)

    trainQ = np.repeat(np.expand_dims(trainQ, axis=1), hops, axis=1)
    validQ = np.repeat(np.expand_dims(validQ, axis=1), hops, axis=1)
    testQ = np.repeat(np.expand_dims(testQ, axis=1), hops, axis=1)
    metadata = {
        "nb_sentences": max_num_sentences,
        "nb_words": max_words,
        "vocab_size": vocab_size,
        "nb_classes": answer_vocab_size if class_space == "answer_corpus" else vocab_size,
        "full_corpus_dict": full_corpus_idx,
        "answer_corpus_dict": answer_corpus_idx,
        "train_size": len(trainS),
        "valid_size": len(validS),
        "test_size": len(testS),
        "task_labels_train": task_labels_train,
        "task_labels_valid": task_labels_valid,
        "task_labels_test": task_labels_test,
    }

    atomic_datasets = []
    for dataset in [(trainS, trainQ, trainA), (validS, validQ, validA), (testS, testQ, testA)]:
        fact_values = np.concatenate((dataset[0], dataset[1]), axis=1)
        fact_types = [0] * dataset[0].shape[1] + [1] * dataset[1].shape[1]
        fact_types = np.tile(fact_types, (dataset[0].shape[0], 1))
        x = (fact_values, fact_types)
        atomic_datasets.append((x, dataset[2]))

    return atomic_datasets[0], atomic_datasets[1], atomic_datasets[2], metadata
