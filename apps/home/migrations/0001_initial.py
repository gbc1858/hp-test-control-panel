# Generated by Django 4.1 on 2023-07-05 16:37

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ProcessData",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("Ch1_MAX", models.TextField()),
                ("Ch2_MAX", models.TextField()),
                ("Ch3_MAX", models.TextField()),
                ("Ch4_MAX", models.TextField()),
                ("record_id", models.IntegerField(unique=True)),
                ("task_id", models.IntegerField()),
                ("comments", models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name="RunTaskLog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("times", models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name="Record",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("type", models.CharField(max_length=2)),
                ("address", models.CharField(max_length=32)),
                ("Xincr", models.TextField()),
                ("NR_Pt", models.TextField()),
                ("Ch1_scale", models.TextField()),
                ("Ch2_scale", models.TextField()),
                ("Ch3_scale", models.TextField()),
                ("Ch4_scale", models.TextField()),
                ("Ch1", models.TextField()),
                ("Ch2", models.TextField()),
                ("Ch3", models.TextField()),
                ("Ch4", models.TextField()),
                ("Ch1_ATTN", models.TextField()),
                ("Ch2_ATTN", models.TextField()),
                ("Ch3_ATTN", models.TextField()),
                ("Ch4_ATTN", models.TextField()),
                ("comments", models.TextField()),
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True, verbose_name="created_at"),
                ),
                (
                    "task",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="home.runtasklog",
                    ),
                ),
            ],
        ),
    ]