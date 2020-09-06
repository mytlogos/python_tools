import logging
import multiprocessing as mp
from datetime import datetime
from logging import handlers
from typing import List, TypedDict, Optional

import numpy as np
from matplotlib.colors import ListedColormap

import pdf.similarity
import pdf.text_process
import pdf.tools
import plots
from pdf.storage import FileStorage, Storage, SqlStorage
from pdf.text_extract import ExtractStage
from pdf.transform import transfer_storage

# 'export' only that one function
__all__ = ["plot", "run"]


class LogMessage(TypedDict):
    message: Optional[str]
    time: Optional[datetime]
    pid: Optional[int]
    stage: Optional[int]
    total: Optional[int]
    current: Optional[int]


class StageLogger(logging.Logger):

    def stage(self):
        pass


def create_logger(queue) -> StageLogger:
    logger = logging.getLogger("pdf")
    logger.setLevel(level=logging.INFO)

    stage_level = logging.INFO + 5

    def stage(self: StageLogger, msg, *args, **kwargs):
        if self.isEnabledFor(stage_level):
            self._log(stage_level, msg, args, **kwargs)

    logger.stage = stage.__get__(logger, None)

    if queue:
        logger.addHandler(handlers.QueueHandler(queue))
    # noinspection PyTypeChecker
    return logger


def run(*, files: List[str] = None, directory: str = None, extensions=None, message_queue=None, config=None):
    tools.init_child_process(mp.Value("i", 0), mp.Value("i", 0))

    logger = create_logger(message_queue)

    extract_stage = ExtractStage(logger, tools.STORAGE.get_text_model())
    process_stage = text_process.ProcessStage(logger, tools.STORAGE.get_text_model(), tools.STORAGE.get_result_model())
    similarity_stage = similarity.SimilarityStage(logger, tools.STORAGE)

    extract_stage.report_progress("Report Work")
    process_stage.report_progress("Report Work")
    similarity_stage.report_progress("Report Work")

    logger.info("Get Config", extra=tools.get_extra())
    config: tools.FileConfig = tools.get_config()
    logger.info("Calculate Necessary Files", extra=tools.get_extra())
    files, total_length = tools.necessary_files(extensions, files, directory)

    if files.values:
        # flatten values to a 1d list
        files = [file for file_list in files.values() for file in file_list]

    logger.info("Update Config", extra=tools.get_extra())
    tools.update_config(config, files)

    logger.info("Extract Text", extra=tools.get_extra())
    extract_stage.sequential(files, config)

    logger.info("Process Text", extra=tools.get_extra())
    process_stage.sequential(files, config)

    logger.info("Calculate Similarity Matrix", extra=tools.get_extra())
    similarity_stage.sequential()
    logger.info("Finished All", extra=tools.get_extra(finished=True))


def compare(*, files: List[str] = None, directory: str = None, extensions=None, text_only=False) -> None:
    # transfer_storage(FileStorage(), SqlStorage())
    # transform_results(ResultSerializer(), LineResultSerializer())
    # get_words_parallel()
    pass
    # if text_only:
    #     do_in_parallel(save_text, files, directory, extensions, unique_words)
    # else:
    #     results = do_in_parallel(get_file_results, files, directory, extensions, unique_words)
    #     analyze(results)


def plot():
    import matplotlib.pyplot as plt

    similarity_matrix = np.load("similarity.npy")

    with np.nditer(similarity_matrix, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for value in it:
            row, column = it.multi_index
            mirror_value = similarity_matrix[column, row]

            if not value and mirror_value:
                similarity_matrix[it.multi_index] = mirror_value
            # if value > 0.5:
            #     print(f"{value} similarity for ({row},{column})")

    similarity_matrix = similarity_matrix[:100, :100]
    row_labels = list(range(similarity_matrix.shape[0]))
    column_labels = list(range(similarity_matrix.shape[1]))
    cmap = ListedColormap(COLOR_MAP)
    plots.heatmap(similarity_matrix, row_labels, column_labels, plt, cmap=cmap)
    plt.gcf().set_size_inches(100, 100)
    plt.legend()
    plt.savefig("test1.svg")
    plt.show()


COLOR_MAP = np.array([
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 254),
    (254, 254, 254),
    (254, 254, 253),
    (253, 254, 253),
    (252, 253, 252),
    (252, 253, 251),
    (251, 253, 251),
    (251, 252, 250),
    (250, 252, 250),
    (249, 251, 249),
    (249, 251, 249),
    (248, 251, 248),
    (247, 250, 248),
    (247, 250, 247),
    (246, 250, 246),
    (246, 249, 246),
    (245, 249, 245),
    (244, 249, 245),
    (244, 248, 244),
    (243, 248, 244),
    (243, 248, 243),
    (242, 247, 242),
    (241, 247, 242),
    (241, 247, 241),
    (240, 242, 239),
    (238, 232, 234),
    (237, 222, 229),
    (235, 212, 224),
    (234, 202, 220),
    (233, 192, 215),
    (231, 182, 210),
    (230, 172, 205),
    (228, 162, 200),
    (227, 152, 196),
    (226, 142, 191),
    (224, 132, 186),
    (223, 122, 181),
    (221, 112, 177),
    (220, 102, 172),
    (219, 92, 167),
    (217, 82, 162),
    (216, 72, 157),
    (214, 62, 153),
    (213, 52, 148),
    (212, 42, 143),
    (210, 32, 138),
    (209, 22, 134),
    (207, 12, 129),
    (206, 2, 124),
    (199, 0, 119),
    (191, 0, 114),
    (183, 0, 109),
    (174, 0, 104),
    (166, 0, 99),
    (158, 0, 94),
    (149, 0, 89),
    (141, 0, 84),
    (133, 0, 79),
    (125, 0, 74),
    (116, 0, 69),
    (108, 0, 65),
    (100, 0, 60),
    (91, 0, 55),
    (83, 0, 50),
    (75, 0, 45),
    (66, 0, 40),
    (58, 0, 35),
    (50, 0, 30),
    (42, 0, 25),
    (33, 0, 20),
    (25, 0, 15),
    (17, 0, 10),
    (8, 0, 5),
    (0, 0, 0)
], dtype=float) / 255
