
##
##      GFPGAN AI
##

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, './ai/gfpgan')
sys.path.insert(0, './ai')

import argparse
import numpy as np
import os
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import cv2
import glob
import torch

gDefault_res = 2
gMax_res = 4
basedirectory = os.path.dirname(__file__)

## where to save the user profile?
def fnGetUserdataPath(_username):
    _path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_PROFILE_DIR = os.path.join(_path, '_profile')
    USER_PROFILE_DIR = os.path.join(DEFAULT_PROFILE_DIR, _username)
    return {
        "location": USER_PROFILE_DIR,
        "voice": False,
        "picture": True
    }

## WARMUP Data
def getWarmupData(_id):
    try:
        import time
        from werkzeug.datastructures import MultiDict
        ts=int(time.time())
        sample_args = MultiDict([
            ('-u', 'test_user'),
            ('-uid', str(ts)),
            ('-t', _id),
            ('-cycle', '0'),
            ('-o', 'warmup.jpg'),
            ('-filename', 'warmup.jpg')
        ])
        return sample_args
    except:
        print("Could not call warm up!\r\n")
        return None
    
def _run(args):
        ## call GFPGAN libs
    try:

        # ------------------------ set up background upsampler ------------------------
        if args.bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                            'If you really want to use it, please modify the corresponding codes.')
                bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=args.bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            bg_upsampler = None

        # ------------------------ set up GFPGAN restorer ------------------------
        if args.version == '1':
            arch = 'original'
            channel_multiplier = 1
            model_name = 'GFPGANv1'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif args.version == '1.2':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANCleanv1-NoCE-C2'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        elif args.version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif args.version == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif args.version == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        elif args.version == 'CodeFormer':
            arch = 'CodeFormer'
            channel_multiplier = 2
            model_name = 'CodeFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth'
        else:
            raise ValueError(f'Wrong model version {args.version}.')
        
        # determine model paths
        model_path = os.path.join(basedirectory, 'experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join(basedirectory, 'gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = url

        restorer = GFPGANer(
            model_path=model_path,
            upscale=args.upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)

        # ------------------------ restore ------------------------
        # read image
        filePathnameIn=os.path.join(args.indir, args.filename)
        filePathnameOut=os.path.join(args.outdir, args.output)
        img_name = os.path.basename(filePathnameIn)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(filePathnameIn, cv2.IMREAD_COLOR)

        basenameOut, extOut = os.path.splitext(os.path.basename(filePathnameOut))

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=args.aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight)

        # save faces
        # for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            # save_crop_path = os.path.join(args.outdir, f'{basenameOut}_{idx:02d}_cropped.png')
            # imwrite(cropped_face, save_crop_path)
            
            # save restored face
            # save_face_name = f'{basenameOut}_{idx:02d}_restored.png'
            # save_restore_path = os.path.join(args.outdir, save_face_name)
            # imwrite(restored_face, save_restore_path)

            # save comparison image
            # cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            # imwrite(cmp_img, os.path.join(args.outdir, f'{basenameOut}_compared.png'))

        # save restored img
        if restored_img is not None:
            if args.ext == 'auto':
                extension = extOut[1:]
            else:
                extension = args.ext

            _resFile=f'{basenameOut}_0.{extension}'
            save_restore_path = os.path.join(args.outdir, _resFile)
            imwrite(restored_img, save_restore_path)
            return _resFile
        
        return None

    except Exception as err:
        print('CRITICAL: Could not run this AI')
        raise err

## ENTRY POINT here
def fnRun(_args): 
    vq_parser = argparse.ArgumentParser()

    # OSAIS arguments
    vq_parser.add_argument("-odir", "--outdir", type=str, help="Output directory", default="./_output/", dest='outdir')
    vq_parser.add_argument("-idir", "--indir", type=str, help="input directory", default="./_input/", dest='indir')

    # Add the GFPGAN arguments
    vq_parser.add_argument('-filename','--filename',type=str,default='warmup.jpg',help='Input image')
    vq_parser.add_argument('-o', '--output', type=str, default='output.jpg', help='Output filename. Default: output.jpg')
    vq_parser.add_argument('-v', '--version', type=str, default='1.3', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    vq_parser.add_argument('-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    vq_parser.add_argument('--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    vq_parser.add_argument('--bg_tile',type=int,default=400,help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    vq_parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    vq_parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    vq_parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    vq_parser.add_argument('--ext',type=str,default='jpg',help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    vq_parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights for CodeFormer.')

    beg_date = datetime.utcnow()
    _resFile=None
    try:
        args = vq_parser.parse_args(_args)
        print(args)
        _resFile=_run(args)

    except Exception as err:
        print("\r\nCRITICAL ERROR!!!")
        raise err

    sys.stdout.flush()
      
    ## return output
    end_date = datetime.utcnow()
    aFile=[]
    if _resFile!=None:
        aFile=[_resFile]
    return {
        "beg_date": beg_date,
        "end_date": end_date,
        "mCost": 1.05,            ## cost multiplier of this AI
        "aFile": aFile
    }
