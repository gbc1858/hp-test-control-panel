from django.db import models


class Record(models.Model):
    type = models.CharField(max_length=2)
    address = models.CharField(max_length=32)
    Xincr = models.TextField()
    NR_Pt = models.TextField()
    Ch1_scale = models.TextField()
    Ch2_scale = models.TextField()
    Ch3_scale = models.TextField()
    Ch4_scale = models.TextField()
    Ch1 = models.TextField()
    Ch2 = models.TextField()
    Ch3 = models.TextField()
    Ch4 = models.TextField()
    Ch1_ATTN = models.TextField()
    Ch2_ATTN = models.TextField()
    Ch3_ATTN = models.TextField()
    Ch4_ATTN = models.TextField()

    comments = models.TextField()

    created_at = models.DateTimeField('created_at', auto_now_add=True)

    task = models.ForeignKey('RunTaskLog', on_delete=models.CASCADE)


class RunTaskLog(models.Model):
    times = models.IntegerField()


class ProcessData(models.Model):
    Ch1_MAX = models.TextField()  # max
    Ch2_MAX = models.TextField()  # max
    Ch3_MAX = models.TextField()  # max
    Ch4_MAX = models.TextField()  # max

    record_id = models.IntegerField(unique=True)
    task_id = models.IntegerField()

    comments = models.TextField()

