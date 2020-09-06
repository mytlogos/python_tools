import cProfile
import json
import math
import multiprocessing as mp
from bisect import bisect_left
from datetime import datetime
from typing import Tuple, Set, Dict, List, Optional, Union, Any

import numpy as np

from pdf.storage import Result, Storage
from pdf.tools import file_results, ResultSerializer, get_data_dir, get_config, STORAGE, init_child_process, \
    get_child_current, get_child_total, get_child_id, FileConfig, Stage, Logger, config_numbers

__all__ = ["SimilarityStage"]

VectorLengthMap = Dict[int, float]
Pair = Tuple[int, int]
ProcessBlocks = List[Optional[List[Pair]]]
ProcessArgs = Tuple[ProcessBlocks, VectorLengthMap]


def calculate_work_blocks(available_results, config: FileConfig):
    data_block: Dict[int, List[Pair]] = dict()
    block_width = 100
    block_number = math.ceil(available_results / block_width)
    available_indices = set(config_numbers(config))

    for index in range(available_results + 1):
        if index not in available_indices:
            continue

        block_y = ((index // block_width) * block_number) + (index // block_width)

        for other_index in range(available_results + 1):
            if other_index not in available_indices or index <= other_index:
                continue

            block_x = ((index // block_width) * block_number) + (index // block_width)
            block_index = block_x + block_y

            data_block.setdefault(block_index, []).append((index, other_index))

    return data_block


# old algorithm for calculating work blocks, maybe uncomment later and use it for optional config
# def calculate_work():
# for i in config.values():
#     process_number: int = round(i / block_number) % (n * 1000)
#     args: Optional[ProcessArgs] = data_block[process_number]
#
#     if not args:
#         blocks: ProcessBlocks = [None] * 1000
#         data_block[process_number] = (blocks, vector_lengths)
#     else:
#         blocks: ProcessBlocks = args[0]
#
#     for j in config.values():
#         block_number = round(j / block_number) % 1000
#
#         block = blocks[block_number]
#
#         if block is None:
#             blocks[block_number] = block = []
#
#         if j < i:
#             block.append((i, j))
#         else:
#             block.append((-1, -1))

def load_indices_result(data_dir: str, index: int, serializer=None) -> Result:
    result_model = STORAGE.get_result_model()

    if result_model.exists(index):
        result = result_model.get(index)
        result.words = STORAGE.get_map_indices_model().get(index)
        del result.x
        return result


def init(config: FileConfig):
    available_results = 0
    total_words = set()
    cached = dict()
    vector_lengths = dict()

    # get the words among all data (without english stopwords), calculate the vector lengths

    for result in STORAGE.get_result_model().get_many(config_numbers(config)):
        td_if = result.td_if
        del result.x
        words: List[str] = result.words
        total_words.update(words)

        vector_lengths[result.index] = np.linalg.norm(td_if)
        available_results += 1
        cached[result.index] = result

    total_words = sorted(list(total_words))
    calculate_map_indices(cached, total_words)
    return available_results, vector_lengths


def calculate_map_indices(cached: Dict[int, Result], total_words):
    indices_model = STORAGE.get_map_indices_model()

    for result in cached.values():
        words = result.words
        index = result.index

        if indices_model.exists(index):
            continue

        words = {bisect_left(total_words, word): index for index, word in enumerate(words)}
        indices_model.save(index, words)


def get_words_parallel():
    print(f"{datetime.now()} Parallel Calculation of Similarity Matrix (fake)")
    config: FileConfig = get_config()
    last_number = max(map(lambda x: x["index"], config.values()), default=0)

    available_results, vector_lengths = init(config)
    data_block: Dict[int, List[Tuple[int, int]]] = calculate_work_blocks(available_results, config)

    current_work = [(value, vector_lengths) for value in data_block.values()]
    total_work = sum([len(value) for value in data_block.values()])

    total = mp.Value("i", int(total_work))
    current = mp.Value("i", 0)
    block_id = mp.Value("i", 0)

    cpu_count = max(mp.cpu_count() - 1, 1)
    init_args = (current, total, block_id)

    with mp.Pool(cpu_count, initializer=init_child_process, initargs=init_args) as pool:
        start = datetime.now()

        # sequential
        # init_child_process(*init_args)
        # results = [compare_results(work) for work in current_work]

        # parallelize the function with the given pool
        results = pool.map(profile_compare_results, current_work)
        print(f"Time taken: {datetime.now() - start}")

    similarity_matrix = process_similarity(data_block, last_number, results)

    print(similarity_matrix)
    np.save("similarity.npy", similarity_matrix)
    # noinspection PyTypeChecker
    np.savetxt("similarity.txt", similarity_matrix)


def process_similarity(data_block, last_number, results):
    empty_list = []
    # flatten the result list, TODO: apparently result_list can be None??
    results = [result for result_list in results for result in (result_list or empty_list)]
    similarity_matrix = np.zeros((last_number + 1, last_number + 1), dtype=float)
    not_done_work: Set[Tuple[int, int]] = set([pair for block in data_block.values() for pair in block])

    for index in range(last_number + 1):
        similarity_matrix[index, index] = 1

    # every value on the diagonal is always the same, a value of 1
    similarity_model = STORAGE.get_similarity_model()
    similarity_model.save_many([(index, index, 1) for index in range(last_number + 1)])

    similarity_model.save_many(results)
    similarity_model.save_many([(result[1], result[0], result[2]) for result in results])

    for result in results:
        x, y, value = result
        similarity_matrix[x, y] = value
        similarity_matrix[y, x] = value

        not_done_work.discard((x, y))
        not_done_work.discard((y, x))

    pre_calculated = similarity_model.get_many(not_done_work)

    # fill the matrix with calculated values from a previous run
    for result in pre_calculated:
        x, y, value = result

        similarity_matrix[x, y] = value
        similarity_matrix[y, x] = value

    return similarity_matrix


def profile_compare_results(args):
    child_id = get_child_id()

    with child_id.get_lock():
        child_id.value += 1
        block_id = child_id.value
    print(f"Comparing Block {block_id}")

    # profile the statement within the same subprocess
    cProfile.runctx("compare_results(args)", globals(), locals(), f"subprocess-block-{block_id}.pstat")


def compare_results(args) -> List[Tuple[int, int, float]]:
    child_current = get_child_current()
    child_total = get_child_total()

    block, vector_lengths = args

    data_dir = get_data_dir()
    similarity_result = []
    block_lengths = 0
    loaded_data = 0
    processed = 0

    required_results = set()
    current_data = dict()
    block_lengths += len(block)

    similarity_model = STORAGE.get_similarity_model()

    for pair in block:
        if similarity_model.exists(pair):
            continue

        required_results.update(pair)

    for required_index in required_results:
        try:
            current_data[required_index] = load_indices_result(data_dir, required_index)
        except json.decoder.JSONDecodeError as e:
            print(f"Failed for Index {required_index}")
            raise e

    loaded_data += len([value for value in current_data.values() if value])

    for pair in block:
        config_index, other_config_index = pair
        result = current_data.get(config_index)
        other_result = current_data.get(other_config_index)

        # may be None if the result file does not exist (original is not extractable or has no text)
        # or is calculated already
        if not result or not other_result:
            continue

        words = result.words
        words_set = set(words.keys())
        td_if = result.td_if
        other_words = other_result.words
        other_td_if = other_result.td_if

        other_words_set = set(other_words.keys())
        common_words_indices = words_set.intersection(other_words_set)

        indices = np.zeros(len(common_words_indices), dtype=int)
        other_indices = np.zeros(len(common_words_indices), dtype=int)

        for index, word_index in enumerate(common_words_indices):
            result_index = words[word_index]
            other_result_index = other_words[word_index]

            indices[index] = result_index
            other_indices[index] = other_result_index

        product_array: np.ndarray = td_if[indices] * other_td_if[other_indices]
        dot_sum = product_array.sum()

        length_product = vector_lengths[config_index] * vector_lengths[other_config_index]
        cosine_similarity = dot_sum / length_product
        similarity_result.append((config_index, other_config_index, cosine_similarity))

        processed += 1

        if (processed % 1000) == 0:
            with child_current.get_lock():
                child_current.value += 1000
                current_value = child_current.value

            print(f"{datetime.now()} Compared {current_value}/{child_total.value}")

    with child_current.get_lock():
        child_current.value += processed % 1000

    print(f"Work amount: {block_lengths}, Loaded Data {loaded_data}, Calculated Results: {len(similarity_result)}")
    print(f"{datetime.now()} Finished comparing {child_current.value}/{child_total.value}")
    return similarity_result


def parallel():
    pass


class SimilarityStage(Stage):
    def __init__(self, logger: Logger, storage: Storage) -> None:
        super().__init__(logger, "similarity")
        self._storage = storage
        self._total_work = None

    def compute_work(self) -> int:
        if self._total_work is None:
            self._total_work = self._storage.get_similarity_model().compute_work()
        return self._total_work

    def sequential(self):
        self.report_started()
        self.check()
        self.report_finished()

    def check(self):
        config: FileConfig = get_config()
        data_dir = get_data_dir()
        last_number = max(config_numbers(config), default=0)
        similarity_matrix = np.zeros((last_number, last_number), dtype=np.float_)
        vector_lengths: Dict[int, float] = dict()
        available_results = 0
        serializer = ResultSerializer()
        cached: Dict[int, Dict[str, Any]] = dict()
        total_words: Union[List[str], Set[str]] = set()

        # get the words among all data (without english stopwords), calculate the vector lengths
        for result in file_results(config, data_dir, serializer=serializer, cached=cached):
            td_if = result["td_if"]
            del result["x"]
            words = result["words"]
            total_words.update(words)

            config_index = config[result["file"]]["index"]
            vector_lengths[config_index] = np.linalg.norm(td_if)
            available_results += 1

        total_words = sorted(list(total_words))

        for result in cached.values():
            words = result["words"]
            result["words"] = {bisect_left(total_words, word): index for index, word in enumerate(words)}

        del total_words

        # update cached total work
        self._total_work = (available_results * available_results) / 2
        current_work = 0
        work_done: Set[Tuple[int, int]] = set()
        finished_indices: Set[int] = set()

        for config_index, result in cached.items():
            words = result["words"]
            words_set = set(words.keys())
            td_if = result["td_if"]

            for other_config_index, other_result in cached.items():
                if other_config_index in finished_indices:
                    continue

                smaller_index = min(other_config_index, config_index)
                bigger_index = max(other_config_index, config_index)
                current_work_pair = (smaller_index, bigger_index)

                # do not do the same work twice
                if current_work_pair in work_done:
                    continue

                other_words = other_result["words"]
                other_words_set = set(other_words.keys())
                common_words_indices = words_set.intersection(other_words_set)

                dot_sum = 0
                other_td_if = other_result["td_if"]

                for word_index in common_words_indices:
                    result_index = words[word_index]
                    other_result_index = other_words[word_index]
                    dot_sum += other_td_if[other_result_index] * td_if[result_index]

                length_product = vector_lengths[config_index] * vector_lengths[other_config_index]
                similarity_matrix[config_index, other_config_index] = dot_sum / length_product
                current_work += 1
                work_done.add(current_work_pair)

            finished_indices.add(config_index)
            self.report_progress(f"Finished: {config_index}", current=current_work)

        np.save("similarity.npy", similarity_matrix)
        # noinspection PyTypeChecker
        np.savetxt("similarity.txt", similarity_matrix)

    def unchecked(self, *args, **kwargs):
        super().unchecked(*args, **kwargs)

    def parallel(self, *args, **kwargs):
        super().parallel(*args, **kwargs)
