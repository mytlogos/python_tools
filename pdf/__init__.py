import json
import multiprocessing as mp
from datetime import datetime
from typing import List, TypedDict, Optional, Any, Dict

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


def parse_run_config(config: Optional[Dict[str, Any]]) -> tools.RunConfig:
    result = dict(processes=1)

    if not config:
        return tools.RunConfig(**result)

    processes = config.get("processes", 1)

    if isinstance(processes, int) and processes > 0:
        result["processes"] = processes

    print(f"Runconfig: {config}, Parsed: {json.dumps(result)}")
    return tools.RunConfig(**result)


def run(*, files: List[str] = None, directory: str = None, extensions=None, message_queue=None,
        run_config: Optional[Dict[str, Any]] = None):
    tools.init_child_process(mp.Value("i", 0), mp.Value("i", 0))

    run_config: tools.RunConfig = parse_run_config(run_config)
    logger = tools.create_logger(message_queue)

    extract_stage = ExtractStage(logger, run_config)
    process_stage = text_process.ProcessStage(logger, run_config)
    similarity_stage = similarity.SimilarityStage(logger, run_config)

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

    extract_stage.run(files, config, message_queue)
    process_stage.run(files, config, message_queue)
    similarity_stage.run(message_queue)
    logger.info("Finished All", extra=tools.get_extra(finished=True))


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
