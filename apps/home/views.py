import json
import requests
import numpy as np
import pyvisa as visa
from datetime import datetime

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import loader
from django.urls import reverse
from django.shortcuts import render

from apps.utils import get_store, update_store, update_data_process_store, get_data_process_store
from apps.home.scope_control import get_WFM_preamble, get_WFM, get_WFM_advance, preset_scope, pull_wfm_all_ch
from apps.home.data_processing import cal_max_power, cal_convolution, cal_envolope
from apps.home.hardware_settings import *
from apps.home.models import Record, RunTaskLog, ProcessData


def index(request):
    context = {'segment': 'index'}

    data = get_store()
    context['fs'] = data.get('fs', '')
    context['ss'] = data.get('ss', '')

    context['fs_ch1_attn'] = data.get('FS_CH1_ATTN', '')
    context['fs_ch2_attn'] = data.get('FS_CH2_ATTN', '')
    context['fs_ch3_attn'] = data.get('FS_CH3_ATTN', '')
    context['fs_ch4_attn'] = data.get('FS_CH4_ATTN', '')

    context['ss_ch1_attn'] = data.get('SS_CH1_ATTN', '')
    context['ss_ch2_attn'] = data.get('SS_CH2_ATTN', '')
    context['ss_ch3_attn'] = data.get('SS_CH3_ATTN', '')
    context['ss_ch4_attn'] = data.get('SS_CH4_ATTN', '')

    context['comments'] = data.get('comments', '')

    html_template = loader.get_template('home/measurement.html')
    return HttpResponse(html_template.render(context, request))


def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        load_template = request.path.split('/')[-1]

        context['segment'] = load_template
        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:
        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))


def set_address(request):
    print('set_address get data', request.POST)
    data = {
        'fs': request.POST.get('fs'),
        'ss': request.POST.get('ss')
    }
    print('Set address data:', data)

    update_store(data)

    return JsonResponse({'code': 0, 'data': data})


def get_attn(request):
    print('get_attn get ATTN from input', request.POST)
    data = {
        'FS_CH1_ATTN': request.POST.get('FS_CH1_ATTN'),
        'FS_CH2_ATTN': request.POST.get('FS_CH2_ATTN'),
        'FS_CH3_ATTN': request.POST.get('FS_CH3_ATTN'),
        'FS_CH4_ATTN': request.POST.get('FS_CH4_ATTN'),
        'SS_CH1_ATTN': request.POST.get('SS_CH1_ATTN'),
        'SS_CH2_ATTN': request.POST.get('SS_CH2_ATTN'),
        'SS_CH3_ATTN': request.POST.get('SS_CH3_ATTN'),
        'SS_CH4_ATTN': request.POST.get('SS_CH4_ATTN'),

        'comments': request.POST.get('comments'),
    }
    update_store(data)

    return JsonResponse({'code': 0, 'data': data})


def get_data_to_process(request):
    print('get_data_to_process get channels', request.POST)

    pf_ch = request.POST.get('pf_ch')
    pr_ch = request.POST.get('pr_ch')
    task_ids = [int(id_) for id_ in request.POST.getlist('taskIds[]')]

    update_data_process_store({
        'P_for': pf_ch,
        'P_ref': pr_ch,
        'Task_ids': task_ids,
    })

    exists_task_ids = {p['task_id'] for p in ProcessData.objects.values('task_id').distinct()}
    form_task_ids = set(task_ids)
    target_task_ids = form_task_ids - exists_task_ids  # filter out the exited task_id in ProcessData

    records = Record.objects.filter(task_id__in=target_task_ids, type='fs').all()
    for record in records:
        task_id = record.task.id
#         print(record.Ch2_ATTN[:2])
        try:
            ch1_attn_approx = float(record.Ch1_ATTN[:2]) * (-1)
        except:
            ch1_attn_approx = 0
        try:
            ch2_attn_approx = float(record.Ch2_ATTN[:2]) * (-1)
        except:
            ch2_attn_approx = 0
        try:
            ch3_attn_approx = float(record.Ch3_ATTN[:2]) * (-1)
        except:
            ch3_attn_approx = 0
        try:
            ch4_attn_approx = float(record.Ch4_ATTN[:2]) * (-1)
        except:
            ch4_attn_approx = 0

        print(f'APPROX. ATTN: {ch1_attn_approx}, {ch2_attn_approx}, {ch3_attn_approx}, {ch4_attn_approx}')
        try:
            pd = ProcessData(task_id=task_id, record_id=record.id, comments=record.comments)

            pd.Ch1_MAX = cal_max_power(json.loads(record.Ch1), FS_CH1_CABLE_ATTN, ch1_attn_approx)
            pd.Ch2_MAX = cal_max_power(json.loads(record.Ch2), FS_CH2_CABLE_ATTN, ch2_attn_approx)
            pd.Ch3_MAX = cal_max_power(json.loads(record.Ch3), FS_CH3_CABLE_ATTN, ch3_attn_approx)
            pd.Ch4_MAX = cal_max_power(json.loads(record.Ch4), FS_CH4_CABLE_ATTN, ch4_attn_approx)

            pd.save()
        except Exception as e:
            print('process data error', e)

    pds = ProcessData.objects.filter(task_id__in=form_task_ids).all()

    results = {}
    comments = {}

    for pd in pds:
        task_id = pd.task_id

        if task_id not in results:
            results[task_id] = []
        if task_id not in comments:
            comments[task_id] = []

        x = float(getattr(pd, pf_ch + '_MAX'))
        y = float(getattr(pd, pr_ch + '_MAX'))
        saved_comments = getattr(pd, 'comments')

        results[task_id].append((x, y))
        comments[task_id]= saved_comments

    return JsonResponse({'code': 0, 'data': {'x': pf_ch, 'y': pr_ch, 'data': results, 'comments': comments}})


def get_data_to_convolve(request):
    print('get_data_to_convolve get task ID', request.POST)
    DATA_PROCESS_JSON = get_data_process_store()
    pf_ch = DATA_PROCESS_JSON['P_for']
    pr_ch = DATA_PROCESS_JSON['P_ref']

    task_id_to_convolve = request.POST.get('concolutionTaskId')
    update_data_process_store({'Task_id_to_convolve': task_id_to_convolve,})

#     records = Record.objects.filter(task_id__in=task_id_to_convolve, type='fs').all()
    records = Record.objects.filter(task_id__exact=task_id_to_convolve, type__exact='fs').all()
    V_for_record = records.order_by('id')[0]
    V_for_bd = getattr(V_for_record, pf_ch)[1:-1].split(',')
    V_for = [float(i) for i in V_for_bd]
    print('Convolution forward channel: ' + pf_ch)
    time = getattr(V_for_record, 'created_at')
    print('Data timestamp: ' + str(time))

    Xincr = float(getattr(V_for_record, 'Xincr'))
    NR_Pt = float(getattr(V_for_record, 'NR_Pt'))
    comments = getattr(V_for_record, 'comments')

    time, V_ref_predict_env = cal_convolution(V_for, Xincr, NR_Pt)
    convolution_results = (time.tolist(), V_ref_predict_env.tolist())

    results = []
    for record in records:
        V_ref = getattr(record, pr_ch)[1:-1].split(',')
        time_mea, V_ref_env = cal_envolope(V_ref, Xincr, NR_Pt)
        results.append((time_mea.tolist(), V_ref_env.tolist()))
        print(time_mea.tolist()[0], time_mea.tolist()[-1])

    return JsonResponse({'code': 0, 'data': {'V_ref_predict_data': convolution_results, 'V_ref_data': results, 'comments': comments}})

def run_task(request):
    data = get_store()

    fs = data['fs']
    ss = data['ss']
    saved_comments = data['comments']
    times = request.POST.get('times')
    times = int(times)

    task = RunTaskLog(times=times)
    task.save()

    #     import os
    #     from apps.utils import APPS_PATH
    #     with open(os.path.join(APPS_PATH, 'home', 'data.json'), encoding='utf8') as f:
    #         data = json.load(f)
    #
    #     get_WFM = lambda item, keys: {key: item for key in keys}

    store = get_store()
    print('store', store)
    fs_ch_list = ['CH1', 'CH2', 'CH3', 'CH4']
    ss_ch_list = ['CH1', 'CH2', 'CH3', 'CH4']
    fs_preambles = get_WFM_preamble(fs, fs_ch_list)
    ss_preambles = get_WFM_preamble(ss, ss_ch_list)
    fs_api, ss_api, fs_data_len, ss_data_len = preset_scope(fs, ss, fs_preambles, ss_preambles, fs_ch_list, ss_ch_list)

#     name1 = request.POST.get('name1')

    for _ in range(times):
        data_dic_fs, data_dic_ss = pull_wfm_all_ch(fs_api, ss_api, fs_ch_list, ss_ch_list, fs_data_len, ss_data_len)
        try:
            fs_result, fs_xincr, fs_hori_len, fs_ch_scale = get_WFM_advance(data_dic_fs, fs_ch_list,
                                                            fs_preambles)  # call the function in scope_control.py
#
#             fs_result, fs_xincr, fs_hori_len, fs_ch_scale = get_WFM(fs, ['CH1', 'CH2', 'CH3', 'CH4'],
#                                                            fs_preambles)  # call the function in scope_control.py
            Record(
                type='fs', address=fs, task=task,
                Xincr=fs_xincr,
                NR_Pt=fs_hori_len,
                Ch1_scale=fs_ch_scale['CH1'],
                Ch2_scale=fs_ch_scale['CH2'],
                Ch3_scale=fs_ch_scale['CH3'],
                Ch4_scale=fs_ch_scale['CH4'],
                Ch1=fs_result['CH1'][0],
                Ch2=fs_result['CH2'][0],
                Ch3=fs_result['CH3'][0],
                Ch4=fs_result['CH4'][0],
                Ch1_ATTN=data['FS_CH1_ATTN'],
                Ch2_ATTN=data['FS_CH2_ATTN'],
                Ch3_ATTN=data['FS_CH3_ATTN'],
                Ch4_ATTN=data['FS_CH4_ATTN'],

                comments=saved_comments,

            ).save()
        except AttributeError:
            print('FS-AttributeError')

        try:
            ss_result, ss_xincr, ss_hori_len, ss_ch_scale = get_WFM_advance(data_dic_ss, ss_ch_list,
                                                       ss_preambles)
#             ss_result, ss_xincr, ss_hori_len, ss_ch_scale = get_WFM(ss, ['CH1', 'CH2'],
#                                                        ss_preambles)  # call the function in scope_control.py

            Record(
                type='ss', address=ss, task=task,
                Xincr=ss_xincr,
                NR_Pt=ss_hori_len,
                Ch1_scale=ss_ch_scale['CH1'],
                Ch2_scale=ss_ch_scale['CH2'],
                Ch3_scale=ss_ch_scale['CH3'],
                Ch4_scale=ss_ch_scale['CH4'],
                Ch1=ss_result['CH1'][0],
                Ch2=ss_result['CH2'][0],
                Ch3=ss_result['CH3'][0],
                Ch4=ss_result['CH4'][0],
                Ch1_ATTN=data['SS_CH1_ATTN'],
                Ch2_ATTN=data['SS_CH2_ATTN'],
                Ch3_ATTN=data['SS_CH3_ATTN'],
                Ch4_ATTN=data['SS_CH4_ATTN'],

                comments=saved_comments,
            ).save()
            print(f'scope finish    | {datetime.now().time()}')
            print(f'--------------- | ---------------')
        except AttributeError:
            print('SS-AttributeError')

    return JsonResponse({'code': 0, 'data': None})


def get_last_task(request):
    last_task = RunTaskLog.objects.last()
    fs_records = Record.objects.filter(task=last_task, type='fs')[::1]
    ss_records = Record.objects.filter(task=last_task, type='ss')[::1]

    if len(fs_records) > 20:
        print('Too many data recorded. Only display first 20 shots.')
        fs_records = Record.objects.filter(task=last_task, type='fs')[0:20]
        ss_records = Record.objects.filter(task=last_task, type='ss')[0:20]

    fs = [serializer_record(record) for record in fs_records]
    ss = [serializer_record(record) for record in ss_records]
    return JsonResponse({'code': 0, 'data': {
        'fs': fs,
        'ss': ss,
    }})


def serializer_record(record):
    res = {'Ch1': json.loads(record.Ch1), 'Ch2': json.loads(record.Ch2), 'Ch3': json.loads(record.Ch3), 'Ch4': json.loads(record.Ch4)}
#     if record.type == 'ss':
#         res['Ch4'] = json.loads(record.Ch4)
#     if record.type == 'fs':
#     res['Ch3'] = json.loads(record.Ch3)
#     res['Ch4'] = json.loads(record.Ch4)
    return res

