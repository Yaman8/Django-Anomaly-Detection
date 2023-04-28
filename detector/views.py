from django.shortcuts import render, HttpResponse
from django.core.files.storage import default_storage
from .video2frame import to_frames
from .vis import generate_vid
from .frame import extract_snippet

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
        # context={'message':file_name}
        to_frames(url,f.name)
        generate_vid(f.name)
        full_url='http://127.0.0.1:8000/media/'+f.name[:-4]+'_result.mp4'
        print(full_url)
        # extract_snippet('/media/yaman/new-e/Django-Anomaly-Detection/media/video.mp4','00:00:07','00:00:10')

    return render(request, 'index.html', context={'video_url': full_url})
    # filename=f.save()
    # return HttpResponse('Home page')
# def display(request):
#     vid=Video.objects.all()

def extract_snippet(request):
    if request.method=='POST':
        start=request.POST.get('start')
        duration=request.POST.get('duration')
        print("Start:",start)
        print("Duration:",duration)
        return render(request,'index.html')
    else:
        return render(request,'index.html')
