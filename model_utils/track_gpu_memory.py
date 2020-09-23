
import gc
import datetime
import pynvml

import torch
import inspect
import numpy as np

from network.unet import UNet


class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        frame: a frame to detect current py-file runtime
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """
    def __init__(self, frame, detail=True, path='', verbose=False, device=0):
        self.frame = frame
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.gpu_profile_fn = path + f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt'
        self.verbose = verbose
        self.begin = True
        self.device = device

        self.func_name = frame.f_code.co_name
        self.filename = frame.f_globals["__file__"]
        if (self.filename.endswith(".pyc") or
                self.filename.endswith(".pyo")):
            self.filename = self.filename[:-1]
        self.module_name = self.frame.f_globals["__name__"]
        self.curr_line = self.frame.f_lineno

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def track(self):
        """
        Track the GPU memory usage
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.curr_line = self.frame.f_lineno
        where_str = self.module_name + ' ' + self.func_name + ':' + ' line ' + str(self.curr_line)

        with open(self.gpu_profile_fn, 'a+') as f:

            if self.begin:
                f.write(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                        f" Total Used Memory:{meminfo.used/1000**2:<7.1f}Mb\n\n")
                self.begin = False

            if self.print_detail is True:
                ts_list = [tensor.size() for tensor in self.get_tensors()]
                new_tensor_sizes = {(type(x), tuple(x.size()), ts_list.count(x.size()), np.prod(np.array(x.size()))*4/1000**2)
                                    for x in self.get_tensors()}
                for t, s, n, m in new_tensor_sizes - self.last_tensor_sizes:
                    f.write(f'+ | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20}\n')
                for t, s, n, m in self.last_tensor_sizes - new_tensor_sizes:
                    f.write(f'- | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} \n')
                self.last_tensor_sizes = new_tensor_sizes

            f.write(f"\nAt {where_str:<50}"
                    f"Total Used Memory:{meminfo.used/1000**2:<7.1f}Mb\n\n")

        pynvml.nvmlShutdown()


if __name__ == "__main__":
    device = torch.device('cuda:4')

    frame = inspect.currentframe()  # define a frame to track
    gpu_tracker = MemTracker(frame)  # define a GPU tracker

    gpu_tracker.track()  # run function between the code line where uses GPU
    net = UNet().to(device).eval()
    gpu_tracker.track()  # run function between the code line where uses GPU

    dummy_tensor_1 = torch.randn(1, 1, 128, 128, 128).float().to(device)
    gpu_tracker.track()
    dummy_tensor_1 = dummy_tensor_1.cpu()

    dummy_tensor_2 = torch.randn(1, 1, 256, 192, 256).float().to(device)
    gpu_tracker.track()
    dummy_tensor_2 = dummy_tensor_2.cpu()

    torch.cuda.empty_cache()
    gpu_tracker.track()