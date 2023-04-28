import math
import multiprocessing
from multiprocessing import Pool, Manager, Queue
from _queue import Empty
from multiprocessing.pool import ThreadPool
import torch
import glob
import numpy as np
import os
import subprocess

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from .model import generate_model
from .learner import Learner
from PIL import Image
import numpy as np
import torch
import time
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

try:
    import accimage
except ImportError:
    accimage = None


# parser = argparse.ArgumentParser(description='Video Anomaly Detection')
# parser.add_argument('--n', default='/media/yaman/new-e/Major-Project/VIS/video/Assault038_x264', type=str, help='file name')
# args = parser.parse_args()


class ToTensor(object):

    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.UntypedStorage.from_buffer(pic.tobytes(), dtype=torch.uint8))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


#############################################################
#                        MAIN CODE                          #
#############################################################
model = generate_model()  # .cuda()  # feature extrctir
classifier = Learner()  # .cuda()  # classifier

device = 'cuda'

if device == 'cuda':
    model = model.cuda()
    classifier = classifier.cuda()

checkpoint = torch.load(
    'detector\\weight\\RGB_Kinetics_16f.pth', map_location=torch.device(device))
model.load_state_dict(checkpoint['state_dict'])
checkpoint = torch.load(
    'detector\\weight\\ckpt.pth', map_location=torch.device(device))
classifier.load_state_dict(checkpoint['net'])

model.eval()
classifier.eval()

# This part is for using multiple processes to perform our computation
############################################################################################################################################


def proc_initializer(_shared_frames_list, _save_path, _chunk_size, _model, _classifier, _inter_proc_q):
    global vid_frames, save_path
    vid_frames = _shared_frames_list
    save_path = _save_path

    global model, classifier
    model = _model  # copy.copy(_model)
    classifier = _classifier  # copy.copy(_classifier)

    global chunk_size
    chunk_size = _chunk_size

    global inputs
    inputs = torch.Tensor(1, 3, 16, 240, 320)

    global num_frames
    num_frames = len(vid_frames)

    global inter_proc_q
    inter_proc_q = _inter_proc_q


def process_chunk(init_frame_ind):
    # this will be shared by all processes
    global vid_frames
    # these are made available by the initializer for a process/worker
    global inputs, model, classifier
    global chunk_size, num_frames, save_path
    global inter_proc_q

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # y_pred = []
    start = init_frame_ind-16
    if start < 0:
        start = 0
    for i in range(16):
        inputs[:, :, i, :, :] = ToTensor(1)(Image.open(vid_frames[start+i]))
        if init_frame_ind == 0:
            cv_img = cv2.imread(vid_frames[start+i])
            # print(cv_img.shape)
            h, w, _ = cv_img.shape
            cv_img = cv2.putText(cv_img, 'FPS : 0.0, Pred : 0.0', (5, 15),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 240), 2)

            path = save_path+'/'+os.path.basename(vid_frames[start+i])
            cv2.imwrite(path, cv_img)
            inter_proc_q.put(1)
            # print("()()()()one put")

    # start reused
    start = init_frame_ind
    if start == 0:
        start += 16
        # y_pred=[0]*16
    end = init_frame_ind+chunk_size
    if end > num_frames:
        end = num_frames

    y_pred = []
    for i in range(start, end):
        inputs[:, :, :15, :, :] = inputs[:, :, 1:, :, :]
        inputs[:, :, 15, :, :] = ToTensor(1)(Image.open(vid_frames[i]))
        inputs = inputs  # .cuda()
        start = time.time()
        output, feature = model(inputs)
        feature = F.normalize(feature, p=2, dim=1)
        out = classifier(feature)
        y_pred.append(out.item())
        end = time.time()
        FPS = str(1/(end-start))[:5]
        out_str = str(out.item())[:5]
        # print(len(x_value)/len(y_pred))

        cv_img = cv2.imread(vid_frames[i])
        cv_img = cv2.putText(cv_img, 'FPS :'+FPS+' Pred :'+out_str,
                             (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 240), 2)
        if out.item() > 0.4:
            cv_img = cv2.rectangle(cv_img, (0, 0), (w, h), (0, 0, 255), 3)

        path = save_path+'/'+os.path.basename(vid_frames[i])
        cv2.imwrite(path, cv_img)
        inter_proc_q.put(1)
        # print("()()()()one put")

    return y_pred


def update_progress(pbar, total):
    print("in update_progress")
    total_updates = 0
    while total_updates < total:
        # print("in ")
        up_val = 0
        try:
            up_val = inter_proc_q.get(block=False)
        except inter_proc_q.Empty:
            up_val = 0
        pbar.update(up_val)
        pbar.refresh()
        total_updates += up_val
        print("()()()()one got", up_val, "---", total_updates)
    return "Done"


def get_preds():
    # this will be shared by all processes
    global vid_frames
    # these are made available by the initializer for a process/worker
    global inputs, model, classifier
    global chunk_size, num_frames, save_path
    global inter_proc_q

    num_chunks = math.ceil(len(vid_frames)/chunk_size)
    chunk_ids = [i*chunk_size for i in range(num_chunks)]

    with Pool(2, initializer=proc_initializer, initargs=[vid_frames, save_path, chunk_size, model, classifier, inter_proc_q]) as p:
        print("In update preds")
        y_preds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for pred in p.imap(process_chunk, chunk_ids):
            y_preds += pred
        return y_preds


def proc_initializer2(_shared_frames_list, _save_path, _chunk_size, _model, _classifier, _inter_proc_q):
    global vid_frames, save_path
    vid_frames = _shared_frames_list
    save_path = _save_path

    global model, classifier
    model = _model  # copy.copy(_model)
    classifier = _classifier  # copy.copy(_classifier)

    global chunk_size
    chunk_size = _chunk_size

    global inter_proc_q
    inter_proc_q = _inter_proc_q


# The following three classes implement a type of process pool that allows us to further create child process
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.


class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)

##################################################################################################################################################


def generate_vid(vid):
    path = 'media/' + \
        vid[:-4] + '/*'
    # path='/media/yaman/new-e/Major-Project/VIS/video/Explosion001_x264.mp4'+'/*'
    save_path = 'media\\' + \
        vid[:-4] + '_result'
    frames = glob.glob(path)
    # print(img)
    frames.sort()
    count = 0

    segment = len(frames)//16
    x_value = [i for i in range(segment)]

    inputs = torch.Tensor(1, 3, 16, 240, 320)
    x_time = [jj for jj in range(len(frames))]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    chunk_size = 100
    # num_chunks = math.ceil(len(frames)/chunk_size)
    # chunk_ids = [i*chunk_size for i in range(num_chunks)]
    # print(chunk_ids)
    inter_proc_q = Queue()
    pbar = tqdm(total=len(frames))

    predictions = []
    with MyPool(2, initializer=proc_initializer2, initargs=[frames, save_path, chunk_size, model, classifier, inter_proc_q]) as p2:
        result = p2.apply_async(get_preds)

        while not result.ready():
            up_val = 0
            try:
                up_val = inter_proc_q.get(block=False)
            except Empty:
                up_val = 0
            pbar.update(up_val)
            # pbar.refresh()

        predictions = result.get()

    print("saving result video...")
    os.system('ffmpeg -i "%s" "%s"' %
              (save_path+'/%05d.jpg', save_path+'.mp4'))
    # plt.plot(x_time, y_pred)
    # plt.savefig(save_path+'.png', dpi=300)
    # plt.cla()


'''
    y_pred is an array of scores for each frame
    returns [(start,end),...] such that for each (start,end) video_frames[start:end] gives all anamolous frames for that interval
'''


def get_suspc_moments(y_pred, threshold):
    y_pred2 = (np.array(y_pred) > threshold).astype(int)
    y_pred3 = np.array([0, *y_pred2[:-1]])
    y_pred3 -= y_pred2

    starts = np.where(y_pred3 == -1)[0]
    ends = np.where(y_pred3 == 1)[0]

    if y_pred2[-1] == 1:
        ends = np.append(ends, len(y_pred2))

    return zip(starts, ends)
