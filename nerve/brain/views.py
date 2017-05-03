#coding:utf-8
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from core import coreBp as bp
import json
import numpy as np
import os
from django.views.decorators.csrf import csrf_exempt
god=bp.God()

def index(request):
    return HttpResponse(u"欢迎光临!")

def index(request):
    return render(request,'index.html');

def turn_string_into_array(str):
    list = str.split(",")
    array = np.arange(len(list))
    for i in range(0,len(list)-1):
        array[i]=list[i]
    return array

def turn_string_into_2axis_array(str):
    list = str.split(",")
    array = np.arange(len(list))
    for i in range(0,len(list)-1):
        array[i]=list[i]

    array=array[np.newaxis,:]
    return array


@csrf_exempt
def test(request):


    return HttpResponse(
        json.dumps({"result": 9}),
        content_type="application/json"
    )


@csrf_exempt
def create(request):
    req_json=recv_data(request)
    shape=req_json.get("shape")
    shape_array=turn_string_into_array(shape)

    input_number=req_json.get("input_number")
    output_number=req_json.get("output_number")
    brain_id=req_json.get("brain_id")

    god.createByGod(shape_array,input_number,output_number,brain_id)



    return HttpResponse(
        json.dumps({"result": "ok"}),
        content_type="application/json"
    )


@csrf_exempt
def pratice(request):
    req_json=recv_data(request)

    pratice_data_address=str(req_json.get("pratice_data"))
    label_data_address=str(req_json.get("label_data"))



    shape=req_json.get("shape")
    shape_array = turn_string_into_array(shape)

    input_number=req_json.get("input_number")
    output_number=req_json.get("output_number")
    brain_id=req_json.get("brain_id")



    result=god.praticeByGod(pratice_data_address, label_data_address, shape_array, input_number, output_number, brain_id)



    return HttpResponse(
        str(result)
    )


@csrf_exempt
def predict(request):
    req_json=recv_data(request)

    input_string=req_json.get("input_array")
    input_array = turn_string_into_2axis_array(input_string)

    shape=req_json.get("shape")
    shape_array = turn_string_into_array(shape)

    input_number=req_json.get("input_number")
    output_number=req_json.get("output_number")
    brain_id=req_json.get("brain_id")

    result=god.predictByGod(input_array,shape_array,input_number,output_number,brain_id)
    print(result)


    return HttpResponse(
    str(result)[1:len(str(result))-1]
    )

@csrf_exempt
def upload_file(request):
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("file", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("no files for upload!")
        destination = open(os.path.join("G:/workplace/Sound/data/test_sound",myFile.name),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()
        return HttpResponse("upload over!")






def recv_data(request):
    if request.method == 'POST':
        received_json_data = json.loads(request.body.decode())
        return received_json_data
    else:
        return {}


   # return HttpResponse(
   #      json.dumps({"result": str(result)}),
   #      content_type="application/json"
   #  )