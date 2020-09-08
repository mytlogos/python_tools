import json
import multiprocessing as mp
import os
import threading
from collections import Counter
from logging import LogRecord, Handler
from logging.handlers import QueueListener
from typing import Dict, Tuple

import django
import matplotlib.pyplot as plt
import numpy as np
from django.core.serializers.json import DjangoJSONEncoder
from django.forms.models import model_to_dict
from django.http import HttpResponseRedirect, Http404
from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.utils import timezone
from django.views import generic
from wordcloud import WordCloud

import pdf
from pdf import SqlStorage
from pdf.tools import ProcessRecord
from .models import Question, Choice, Task, TaskMessage, TaskStage, Runnable, TaskConfig


class IndexView(generic.ListView):
    template_name = "polls/index.html"
    context_object_name = "latest_question_list"

    def get_queryset(self):
        """
        Return the last five published questions (not including those set to be
        published in the future).
        """
        return Question.objects.filter(
            pub_date__lte=timezone.now()
        ).order_by("-pub_date")[:5]


class DetailView(generic.DetailView):
    model = Question
    template_name = "polls/detail.html"

    def get_queryset(self):
        """
        Excludes any questions that aren"t published yet.
        """
        return Question.objects.filter(pub_date__lte=timezone.now())


class ResultsView(generic.DetailView):
    model = Question
    template_name = "polls/results.html"


def home(request):
    return render(request, "polls/home.html", context={"tools": [
        {"title": "Library", "name": "library"},
        {"title": "Library Calculator", "name": "library-calculator"}
    ]})


def library(request):
    config = load_config()
    dir_tree = dict()

    for file in config:
        paths = []
        parent = os.path.dirname(file)

        while parent != file:
            next_parent, name = os.path.split(parent)

            if name:
                paths.insert(0, name)

            parent, file = next_parent, parent

        drive, _ = os.path.splitdrive(file)
        contents = dir_tree.setdefault(drive, dict())

        for name in paths:
            contents = contents.setdefault(name, dict())

    return render(request, "polls/library.html", context={"config": config, "dir_tree": dir_tree})


def library_calculator(request):
    tasks = Task.objects.all()
    return render(request, "polls/library-calculator.html", context={"tasks": tasks})


def library_calculator_detail(request, pk):
    task = get_object_or_404(Task, pk=pk)
    return render(request, "polls/library-calculator-detail.html", context={"task": task})


def create_task(request):
    task = Task.objects.create()
    TaskConfig.objects.create(task=task)
    return redirect("polls:library-calculator-detail", pk=task.pk)


class QueueHandler(Handler):

    def __init__(self, task_key: int) -> None:
        super().__init__()
        self._buffer = []
        self._current_suffix = "_current"
        self._total_suffix = "_total"
        self._task_key = task_key
        self.task = Task.objects.get(pk=self._task_key)

    def handle(self, record: ProcessRecord):
        self._buffer.append(record)

        if timezone.is_naive(record.date_time):
            record.date_time = timezone.make_aware(record.date_time)

        if record.stage:
            self.update_stage(record, record.stage, record.current, record.total)

        # if enough records are amassed (arbitrary limit), save them to database
        if len(self._buffer) > 100:
            self.flush()

    def flush(self) -> None:
        messages = []
        for record in self._buffer:
            try:
                reason = record.reason
            except AttributeError:
                reason = None

            messages.append(TaskMessage(current=record.current, total=record.total, level=record.levelname,
                                        content=record.message, date_time=record.date_time, reason=reason,
                                        task=self.task, pid=record.pid, stage=record.stage, state=record.state))

        TaskMessage.objects.bulk_create(messages)
        self._buffer = []

    def update_stage(self, record, stage_name, current, total):
        try:
            task_stage = self.task.taskstage_set.get(name=stage_name)
            task_stage.current = current
            task_stage.total = total
        except TaskStage.DoesNotExist:
            task_stage = TaskStage(current=current, total=total, start_time=record.date_time,
                                   task=self.task, name=stage_name)

        if record.state == Runnable.SUCCEEDED:
            task_stage.state = Runnable.SUCCEEDED
            task_stage.end_time = record.date_time

        elif record.state == Runnable.NONE:
            task_stage.state = Runnable.NONE

        elif record.state == Runnable.RUNNING:
            if task_stage.state != Runnable.RUNNING:
                task_stage.start_time = timezone.now()
            task_stage.state = Runnable.RUNNING

        elif record.state == Runnable.FAILED:
            task_stage.state = Runnable.FAILED
            task_stage.end_time = record.date_time

        elif record.state == Runnable.CANCELED:
            task_stage.state = Runnable.CANCELED
            task_stage.end_time = record.date_time
        task_stage.save()


class ThreadSafeDict(dict):
    """
    Enables the Usage of this Dict as a ContextManager for
    synchronizing across threads.

    Copied from: https://stackoverflow.com/a/29532297
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self._lock.release()


calculator_processes: Dict[int, Tuple[mp.Process, QueueListener, QueueHandler]] = ThreadSafeDict()
calculator_threads: Dict[int, threading.Thread] = ThreadSafeDict()


class LogEncoder(DjangoJSONEncoder):

    def default(self, o):
        if isinstance(o, LogRecord):
            return o.__dict__
        return super().default(o)


def monitor_process(task_key):
    """
    Wait for the Completion of the Process associated with the task_key.
    Updates the Task afterwards and removes this Thread from the thread container 'calculator_threads'.

    :param task_key: the primary key of a Task
    :return: None
    """
    try:
        with calculator_processes:
            process, listener, handler = calculator_processes[task_key]
        process.join()

        with calculator_processes:
            values = calculator_processes.pop(task_key, None)

        exit_code = process.exitcode
        process.close()
        listener.stop()
        handler.flush()
        handler.close()

        # if no values available, someone else already updated task and removed the mapping
        if not values:
            return

        task = Task.objects.get(pk=task_key)

        if not task.is_finished():
            task.state = Task.SUCCEEDED if exit_code == 0 else Task.FAILED
            task.end_time = timezone.now()
            task.save()
    finally:
        print(f"Monitor thread for Task {task_key} finished")

        # clean itself up after finishing
        with calculator_threads:
            calculator_threads.pop(task_key)


def api_progress(request, pk):
    if request.method == "GET":
        values = list(TaskStage.objects.filter(task_id=pk).values())
        return django.http.JsonResponse(values, safe=False)
    else:
        return django.http.HttpResponseNotAllowed(["GET"])


def api_task(request, pk):
    if request.method == "GET":
        task = get_object_or_404(Task, pk=pk)
        value = model_to_dict(task)
        value["stages"] = list(TaskStage.objects.filter(task_id=pk).values())
        value["config"] = model_to_dict(task.taskconfig) if task.taskconfig else None
        return django.http.JsonResponse(value, safe=False)
    elif request.method == "DELETE":
        task = get_object_or_404(Task, pk=pk)
        task.delete()
        return django.http.JsonResponse({"message": "success"})
    else:
        return django.http.HttpResponseNotAllowed(["GET"])


def api_messages(request, pk):
    if request.method == "GET":
        values = list(TaskMessage.objects.filter(task_id=pk).values())
        return django.http.JsonResponse(values, safe=False)
    else:
        return django.http.HttpResponseNotAllowed(["GET"])


def api_start(request: django.http.HttpRequest, pk):
    if request.method == "POST":
        task = get_object_or_404(Task, id=pk)

        if task.is_finished():
            return django.http.JsonResponse({"msg": "Task already finished"})

        if pk in calculator_processes or task.is_running():
            return django.http.JsonResponse({"msg": "Existing Process already running"})

        config = json.loads(request.body.decode("utf8"))

        if config["processes"] >= 1:
            task_config, _ = TaskConfig.objects.get_or_create(task_id=pk)
            task_config.processes = config["processes"]
            task_config.save()

        # do not limit the number of messages (at the cost of a possible OOM Error)
        queue = mp.Queue(-1)
        process = mp.Process(
            target=pdf.run,
            kwargs={"directory": "D:\\Bücher\\", "message_queue": queue, "run_config": config}
        )
        handler = QueueHandler(task_key=pk)
        queue_listener = QueueListener(queue, handler)
        queue_listener.start()

        task.state = Task.RUNNING
        task.start_time = timezone.now()
        task.save()

        process.start()
        task.pid = process.pid
        task.save()

        with calculator_processes:
            calculator_processes[pk] = (process, queue_listener, handler)

        thread = threading.Thread(target=monitor_process, args=(pk,))

        with calculator_threads:
            calculator_threads[pk] = thread
        thread.start()

        return django.http.JsonResponse({"msg": "Process started", "pid": process.pid})
    else:
        return django.http.HttpResponseNotAllowed(["POST"])


def api_stop(request, pk):
    if request.method == "POST":
        task = get_object_or_404(Task, id=pk)

        with calculator_processes:
            values = calculator_processes.pop(pk, None)

        if not values:
            raise Http404("Task does not exist or does not have any processes associated")

        task.state = Task.CANCELED
        task.end_time = timezone.now()
        task.save()

        process, _, _ = values
        process.terminate()
        return django.http.JsonResponse({"msg": "Process stopped", "pid": process.pid, "alive": process.is_alive()})
    else:
        return django.http.HttpResponseNotAllowed(["POST"])


def library_detail(request, book_index: int):
    png = f"./polls/static/polls/images/{book_index}.png"
    result = SqlStorage().get_result_model().get(book_index)
    sorted_indices = np.flip(np.argsort(result.x))
    table = []

    for sorted_index in sorted_indices:
        table.append((result.words[sorted_index], result.x[sorted_index], result.td_if[sorted_index]))

    if not os.path.exists(png):
        print(f"Does not exist: Relative '{png}', Absolute '{os.path.abspath(png)}'")
        if result:
            mapping = dict()
            for word_index, word in enumerate(result.words):
                frequency = result.x[word_index]
                mapping[word] = frequency
            counter = Counter()
            counter.update(mapping)

            wordcloud = WordCloud(background_color='white',
                                  width=1200,
                                  height=1000
                                  ).generate_from_frequencies(counter)
            plt.imshow(wordcloud)
            figure: plt.Figure = plt.gcf()
            figure.set_figwidth(10)
            figure.set_figheight(10)
            plt.savefig(png)
            plt.close()
    return render(
        request,
        "polls/library-detail.html",
        context={
            "index": book_index,
            "image": f"/static/polls/images/{book_index}.png",
            "table": table
        }
    )


def index(request):
    latest_question_list = Question.objects.order_by("-pub_date")[:5]
    context = {"latest_question_list": latest_question_list}
    return render(request, "polls/index.html", context)


def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, "polls/detail.html", {"question": question})


def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, "polls/results.html", {"question": question})


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST["choice"])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, "polls/detail.html", {
            "question": question,
            "error_message": "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse("polls:results", args=(question.id,)))


def load_config() -> Dict[str, int]:
    from pdf.tools import get_config
    return get_config()
