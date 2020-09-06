import re
from logging import Logger
from typing import Tuple, Optional

import pdfminer.high_level as miner
import textract

from pdf.storage import TextModel
from pdf.tools import get_child_current, get_child_total, Stage

__all__ = ["ExtractStage"]


class ExtractStage(Stage):

    def __init__(self, logger: Logger, model: TextModel) -> None:
        super().__init__(logger, "extract")
        self._model = model
        self.child_current = get_child_current()
        self.child_total = get_child_total()
        self._total_work = None

    def compute_work(self) -> int:
        if self._total_work is None:
            self._total_work = self._model.compute_work()
        return self._total_work

    def sequential(self, files, config):
        self._total_work = len(files)
        self.report_started()

        for index, file in enumerate(files):
            file_config = config[file]
            self.check((file, file_config["index"]))
            self.report_progress(f"Extracted from {file}", current=index + 1)

        self.report_finished()

    def check(self, args: Tuple[str, int], *, return_data=False) -> Optional[str]:
        file, config_index = args

        if self._model.exists(config_index):
            if return_data:
                return self._model.get(config_index)
        else:
            value = self.unchecked(args, return_data=return_data)

            if return_data:
                return value

    def unchecked(self, args: Tuple[str, int], *, return_data=False) -> Optional[str]:
        file, config_index = args

        # extract the purely the text without any formatting
        try:
            if file.endswith(".pdf"):
                text = miner.extract_text(file)
            else:
                text = textract.process(file)
        except Exception as e:
            self.report_warning(f"Failed extracting from {file}", reason=e)
            return

        # ensure it is a string
        if not isinstance(text, str):
            text = str(text, "utf8")

        # unescape newlines
        pattern = re.compile("(\\+r)|(\\+n)|(\\+r\\+n)")
        sub = pattern.sub("\n", text).strip()

        self._model.save(config_index, sub)

        if not sub:
            self.report_warning("No data for " + file)

        if return_data:
            return sub

    def parallel(self):
        # todo implement
        super().parallel()
