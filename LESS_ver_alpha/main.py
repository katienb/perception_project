import argparse
import psutil
import os
from utils import set_seed, print_info, get_shape_dtype_itemsize, merge_groups, split_indices
import numpy as np
import torch
from less import denoise
import tifffile
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
import torch.multiprocessing as mp


def denoise_worker(cuda_id, chunk, index, kwargs, output_queue):
    with torch.cuda.device(cuda_id):
        kwargs['verbose_prefix'] = f'Chunk: {index}, '
        result = denoise(chunk.cuda(), **kwargs)
    output_queue.put((index, result.cpu().numpy()))

def main(args):
    # load 
    # seed = args.seed
    # data = args.data
    # gt = args.gt
    # save_dir = args.save_dir
    # cuda = args.cuda
    # patch_size = args.patch_size
    # top_k = args.top_k
    # window_size = args.window_size
    # stride = args.stride
    # temp_size = args.temp_size
    # temp_overlap = args.temp_overlap
    # spat_width = args.spat_width
    # spat_height = args.spat_height
    # spat_overlap = args.spat_overlap
    # verbose = args.verbose
    # debug = args.debug
    set_seed(args.seed)

    # args.data = '/mnt/nvme1n1/zpyan/data/neuron/CaImAn/demoMovie.tif'
    # args.data = "/mnt/nvme2n1/zpyan/projects/LESS/python/DEE518A0.tif"
    # args.data = "/mnt/nvme2n1/zpyan/projects/LESS/python/Barbara.tif"
    # args.data = "/mnt/nvme1n1/zpyan/data/neuron/cityu210/data_real_more/Raw/DeepImaging/mG6f7mm01/20231215/meas06/mG6f7mm01_d231215_s06_00001.tif"
    # args.data = "/mnt/nvme1n1/zpyan/data/neuron/cityu210/data_real_more/Raw/UltralowPower/mG6f7mm01/20231215/meas06/mG6f7mm01_d231215_s06_00001.tif"
    # args.verbose = True
    # args.debug = True
    # 

    if args.data is None:
        print('args.data is None, program exit.')
        return

    if not os.path.exists(args.data):
        print(f'args.data: {args.data} does not exist, program exit.')
        return
    
    if not (args.data.endswith('.tif') or args.data.endswith('.tiff')):
        print(f'args.data: {args.data} is not a TIFF file, program exit.')
        return
    
    # if args.gt is not None:
    #     if not os.path.exists(args.gt):
    #         print(f'args.gt: {args.gt} does not exist, program exit.')
    #         return
        
    #     if not (args.gt.endswith('.tif') or args.gt.endswith('.tiff')):
    #         print(f'args.gt: {args.gt} is not a TIFF file, program exit.')
    #         return
        
    # if args.split_factor is not None:
    #     if args.split_factor < 2 or args.split_factor % 2 != 0:
    #         print("args.split_factor must be an even number greater than or equal to 2. Program exit.")
    #         return

    if args.debug:
        print_info(args)

    cpu_mem_avail_size = psutil.virtual_memory().available

    if args.verbose:
        print(f"Available CPU memory: {cpu_mem_avail_size / 1024**3:.2f} GB")

    # try fit entire first
    data_shape, data_dtype, data_itemsize = get_shape_dtype_itemsize(args.data)

    if args.debug:
        print(f'data, shape: {data_shape}, dtype: {data_dtype}, itemsize: {data_itemsize}')

    if data_shape[0] == 1:
        print(f'single frame data is not supported. Program exit.')
        return
    
    # Estimate memory usage of the data file in bytes
    # data_shape: (frames, height, width)
    # Check dtype; default to uint16 (2 bytes), use 4 bytes if np.float
    # dtype_bytes = 2  # default uint16
    # dtype_bytes = 4 # float
    data_mem = np.prod(data_shape) * data_itemsize

    if cpu_mem_avail_size < data_mem:
        print(f"Not enough CPU memory to load data, required {data_mem / 1024**3:.2f} GB, available {cpu_mem_avail_size / 1024**3:.2f} GB. Program exit.")
        return 
    
    if args.debug:
        print(f"CPU memory usage of data: {data_mem / 1024**3:.2f} GB")
    
    # load data
    data = tifffile.imread(args.data)
    dtype = data.dtype
    data = torch.from_numpy(data).float()

    # data = torch.ones([100000, 512, 512])
    # data = torch.ones([100000, 1024, 1024])
    # data = torch.ones([200000,  512, 512])
    # data = torch.ones([200000, 1024, 1024])

    # Calculate total bytes of the loaded data tensor
    data_bytes = data.numel() * data.element_size()
    if args.debug:
        print(f"Loaded data occupies {data_bytes / 1024**3:.2f} GB in memory")

    c, h, w = data.shape
    m = h * w

    bytes_per_pixel = data.element_size()
    
    # SVD result + original data
    peak_data_bytes = (2*m*c + 2*c*c) * bytes_per_pixel * 1.5
    if args.debug:
        print(f"Peak mem occupies {peak_data_bytes / 1024**3:.2f} GB in memory")

    # check gpu
    if not args.cpu:
        if torch.cuda.is_available():
            nvmlInit()
            num_devices = torch.cuda.device_count()
            if args.debug:
                print(f"Number of CUDA devices: {num_devices}")

            # device_indices = list(range(num_devices))
            free_cuda_mem_list = []
            for device_id in range(num_devices):
                handle = nvmlDeviceGetHandleByIndex(device_id)
                mem_info = nvmlDeviceGetMemoryInfo(handle)

                free_cuda_mem = mem_info.free
                if args.debug:
                    print(f"   [CUDA: {device_id}] Free Memory: {free_cuda_mem / (1024 ** 3):.2f} GB")

                free_cuda_mem_list.append(free_cuda_mem)
            nvmlShutdown()

            max_cuda_mem = np.max(free_cuda_mem_list)
            single_gpu = False
            if max_cuda_mem >= peak_data_bytes:
                # single gpu
                single_gpu = True
                cuda_id = np.argmax(free_cuda_mem_list)
                if args.debug:
                    print(f"Use CUDA: {device_id} with max avaiable CUDA Free Memory: {max_cuda_mem / (1024 ** 3):.2f} GB")

            else:
                # split with multi gpu
                # Split data spatially (on H, W) based on minimum available GPU memory
                min_free_mem = np.min(free_cuda_mem_list)
                # Estimate how many pixels fit into min_free_mem (use 80% as margin)
                safe_mem = min_free_mem * 1.0
                bytes_per_pixel = data.element_size()
                c, h, w = data.shape
                # For each chunk, we need to store input and output, so double the memory

                # peak_data_bytes = (2*m*c + 2*c*c) * data.element_size() * 1.5

                # chunk_peak_data_bytes = None
                s = 2
                exist = False
                while True:
                    _chunk_peak_data_bytes = 0
                    for offset_h in range(s):
                        for offset_w in range(s):
                            # Generate indices
                            h_idx = np.arange(offset_h, h, s)
                            w_idx = np.arange(offset_w, w, s)
                            
                            _current_peak_data_bytes = (2* len(h_idx) * len(w_idx) * c + 2*c*c) * bytes_per_pixel * 1.5

                            if _current_peak_data_bytes > _chunk_peak_data_bytes:
                                _chunk_peak_data_bytes = _current_peak_data_bytes

                    if safe_mem >= _chunk_peak_data_bytes:
                        break
                    if args.debug:
                        print(f's: {s}')

                    if s > max(h, w):
                        exist = True
                        break
                    s += 1
                
                if exist:
                    print(f'Cannot process this data. Program exit.')
                    return

                # groups, h_indices, w_indices = split_data(data=data, s=s)

                h_indices, w_indices, indices = split_indices(c=c, h=h, w=w, s=s)
                # print(indices)

                if args.debug:
                    print(f'Split data with step size: {s}, chunk num.: {len(indices)}, each chunk occupies {_chunk_peak_data_bytes / (1024 ** 3):.2f} GB in memory.')

        else:
            print("CUDA is not available, running on CPU.")
            args.cpu = True
    
    if args.cpu:
        # cpu only
        denoised = denoise(data=data, cuda=False,
                            pat=args.pat,
                            patch_size=args.patch_size,
                            top_k=args.top_k,
                            window_size=args.window_size,
                            stride=args.stride).numpy()
        
    else:
        # gpu
        ## processing logic
        ### if the data can be handled by a single GPU, then just use a single GPU
        ### if not, try to split it
        if single_gpu:
            with torch.cuda.device(cuda_id):
                denoised = denoise(data=data.cuda(), cuda=True,
                            pat=args.pat,
                            patch_size=args.patch_size,
                            top_k=args.top_k,
                            verbose=args.verbose,
                            window_size=args.window_size,
                            stride=args.stride).cpu().numpy()
        else:
            # mult gpu
            mp.set_start_method('spawn', force=True)
            
            # Assign each chunk to a device in round-robin fashion
            chunk_device_ids = [i % num_devices for i in range(len(indices))]
            
            kwargs = dict(
                cuda=True,
                pat=args.pat,
                patch_size=args.patch_size,
                top_k=args.top_k,
                window_size=args.window_size,
                verbose=args.verbose,
                stride=args.stride
            )

            output_queue = mp.Queue()

            processes = []
            for chunk_cuda_id, idx, h_idx, w_idx in zip(chunk_device_ids, indices,  h_indices, w_indices):
                p = mp.Process(target=denoise_worker, args=(chunk_cuda_id, data[:, h_idx[:, None], w_idx], idx, kwargs, output_queue))
                p.start()
                processes.append(p)

            results = [None] * len(chunk_device_ids)
            for _ in range(len(chunk_device_ids)):
                index, result = output_queue.get()
                results[index] = result

            for p in processes:
                p.join()

            # Merge results back to full image
            if args.debug:
                print(f'Finish processing, merge denoised chunks...')

            denoised = merge_groups(results, h, w, h_indices, w_indices)

    # return

    # save data
    data_name, data_suffix = os.path.splitext(os.path.basename(args.data))
    if args.save_dir is None:
        save_path = os.path.join(os.path.dirname(args.data), f"{data_name}_denoised_LESS{data_suffix}")
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f'{data_name}_denoised_LESS{data_suffix}')

    # save
    tifffile.imwrite(save_path, denoised.astype(dtype))
        


    







# def less(filepath:str,
#          filepath_gt: Optional[str] = None,
#          chunk_size: Optional[int] = 2000,
#          device: Optional[str] = None,
#          *args, **kwargs) -> np.ndarray:
#     """
#     Main function to run LESS denoising on a TIFF file.
    
#     Args:
#         filepath (str): Path to the TIFF file.
#         *args: Additional positional arguments.
#         **kwargs: Additional keyword arguments.
        
#     Returns:
#         None
#     """
#     # check input arguments
#     if chunk_size is not None:
#         assert isinstance(chunk_size, int) and chunk_size > 1, "chunk_size must be a positive integer greater than 1"
#     assert os.path.exists(filepath), "File does not exist"
#     assert filepath.endswith('.tif') or filepath.endswith('.tiff'), "File must be a TIFF file"

#     # set device
#     if device is not None:
#         assert isinstance(device, str) and (device.startswith('cuda') or device == 'cpu'), "device must be 'cuda', 'cuda:i' or 'cpu'"
#         device = torch.device(device)
#     else:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # get shape of the TIFF file
#     shape = get_shape(filepath)
#     max_num_frames = shape[0]

#     # set chunk start and end indices



#     # load frames by chunk
#     if chunk_size is not None:
#         assert isinstance(chunk_size, int) and chunk_size > 0, "chunk_size must be a positive integer"
#         num_frames = shape[0]
#         chunks = [load_frames(filepath, i, min(i + chunk_size, num_frames)) for i in range(0, num_frames, chunk_size)]
#     else:
#         chunks = [load_frames(filepath)]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LESS Denoising (version: alpha)')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    
    # data
    parser.add_argument('--data', type=str)
    # parser.add_argument('--gt', type=str)
    parser.add_argument('--save_dir', type=str)

    # default cuda runner
    parser.add_argument('--cpu', action='store_true')
    
    # LESS cfgs
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--top_k', type=int, default=18)
    parser.add_argument('--window_size', type=int, default=37)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--pat', type=int, default=5)

    # split data
    # parser.add_argument('--auto_split', action='store_true')
    # parser.add_argument('--split_factor', type=int)

    # print info
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    main(args)
