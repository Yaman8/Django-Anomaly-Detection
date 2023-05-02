from django.shortcuts import render, HttpResponse
from django.core.files.storage import default_storage
from .video2frame import to_frames
from .vis import generate_vid
from .vis import get_suspc_moments
from .frame import extract_snippet as other_extract_snippet
import json


# Create your views here.


def home(request):
    return render(request, 'index.html')


def save_video(request):
    print('--->> hoi hi')
    if request.method == 'POST':
        f = request.FILES['sentFile']
        print('*********', f.name)
        file_name = default_storage.save(f.name, f)
        url = default_storage.url(file_name)
        context = {'message': file_name}
        to_frames(url, f.name)
        # susp_moments,thumbs, y_preds = generate_vid(f.name)
        susp_moments, y_preds = generate_vid(f.name)
        susp_moments = list(susp_moments)
        # susp_moments = list(susp_moments)
        # print(type(susp_moments))
        global full_url

        save_path = 'media\\' + \
            f.name[:-4] + '_result'
        vid_save_path = save_path+'_video'
        full_url = 'http://127.0.0.1:8000/'+vid_save_path+'/video.mp4'
        # print('y_preds',y_preds)
    return render(request, 'index.html', context={'video_url': full_url, 'susp_moments': (json.dumps(susp_moments)), 'y_preds': (json.dumps(y_preds))})
    # filename=f.save()
    # return HttpResponse('Home page')
# def display(request):
#     vid=Video.objects.all()


def extract_snippet(request):
    if request.method == 'POST':

        # print(request.POST)
        # start = request.POST.get('start')
        # duration = request.POST.get('duration')

        post_data = json.loads(request.body.decode("utf-8"))
        start = post_data.get('start')
        duration = post_data.get('duration')
        print("Start:", start)
        print("Duration:", duration)
        global full_url

        my_url = full_url.split('/', 3)[3]
        print(full_url)
        other_extract_snippet(
            my_url, start, duration)
        return render(request, 'index.html')
    else:
        return render(request, 'index.html')
