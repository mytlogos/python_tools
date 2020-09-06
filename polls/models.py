import datetime

from django.db import models
from django.utils import timezone


# Create your models here.


class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.question_text

    def was_published_recently(self):
        now = timezone.now()
        # noinspection PyTypeChecker
        return now - datetime.timedelta(days=1) <= self.pub_date <= now


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text


class Runnable(models.Model):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    FAILED = "failed"
    NONE = "none"
    task_states = [
        (NONE, "None"),
        (RUNNING, "Running"),
        (FAILED, "Failed"),
        (CANCELED, "Canceled"),
        (SUCCEEDED, "Finished"),
    ]
    start_time = models.DateTimeField(null=True)
    end_time = models.DateTimeField(null=True)
    state = models.CharField(choices=task_states, default=NONE, max_length=200)

    def is_running(self):
        return self.state == self.RUNNING

    def is_failed(self):
        return self.state == self.FAILED

    def is_canceled(self):
        return self.state == self.CANCELED

    def is_succeeded(self):
        return self.state == self.SUCCEEDED

    def is_finished(self):
        return self.state in {self.FAILED, self.CANCELED, self.SUCCEEDED}

    def is_none(self):
        return self.state == self.NONE

    class Meta:
        abstract = True


class Task(Runnable):
    pid = models.IntegerField(null=True)

    def __str__(self):
        return str(self.pid)


class TaskConfig(models.Model):
    task = models.OneToOneField(Task, on_delete=models.CASCADE)
    processes = models.IntegerField(default=1)

    def __str__(self):
        return f"Config of Task with Id{self.task}"


class TaskStage(Runnable):
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    current = models.IntegerField(default=0)
    total = models.IntegerField(default=0)


class TaskMessage(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    content = models.TextField()
    reason = models.TextField(null=True)
    pid = models.IntegerField()
    stage = models.CharField(max_length=200, null=True)
    level = models.CharField(max_length=200)
    current = models.IntegerField()
    total = models.IntegerField()
    state = models.CharField(choices=Runnable.task_states, default=Runnable.NONE, max_length=200, null=True)

    def __str__(self):
        return self.content
