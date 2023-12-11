import sys
# sys.path.append('../')
sys.path.append('./')

import imageio
import lpips
from tqdm import tqdm

from torch.utils.data import DataLoader

from config import config_parser
from gnesf.sample_ray import RaySamplerSingleImage
from gnesf.render_image import render_single_image
from gnesf.model import IBRNetModel
from utils import *
from gnesf.projection import Projector
from gnesf.data_loaders import dataset_dict
from gnesf.ssim_torch import ssim as ssim_torch


mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''

    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def img2ssim(gt_image, pred_image):
    """
    Args:
        gt_image: [B, 3, H, W]
        pred_image: [B, 3, H, W]
    """
    return ssim_torch(gt_image, pred_image).item()


def img2lpips(lpips_loss, gt_image, pred_image):
    return lpips_loss(gt_image * 2 - 1, pred_image * 2 - 1).item()


def compose_state_dicts(model) -> dict:
    state_dicts = dict()
    
    state_dicts['net_coarse'] = model.net_coarse
    state_dicts['feature_net'] = model.feature_net
    if model.net_fine is not None:
        state_dicts['net_fine'] = model.net_fine

    return state_dicts


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    print(args)
    args.distributed = False
    
    mode = args.mode

    # Create IBRNet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    state_dicts = compose_state_dicts(model=model)

    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}/eval/'.format(args.rootdir, args.expname)
    print("saving results to eval/{}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    print(args.eval_scenes)
    if len(args.eval_scenes) == 1: print('Scene: {}'.format(args.eval_scenes[0]))

    test_dataset = dataset_dict[args.eval_dataset](args, mode, scenes=args.eval_scenes)
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    results_dict = {} # scene_name: {}
    sum_coarse_psnr = 0
    sum_fine_psnr = 0
    running_mean_coarse_psnr = 0
    running_mean_fine_psnr = 0
    sum_coarse_lpips = 0
    sum_fine_lpips = 0
    running_mean_coarse_lpips = 0
    running_mean_fine_lpips = 0
    sum_coarse_ssim = 0
    sum_fine_ssim = 0
    running_mean_coarse_ssim = 0
    running_mean_fine_ssim = 0

    lpips_loss = lpips.LPIPS(net="alex").cuda()

    for i, data in enumerate(tqdm(test_loader)):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()

        items = data['rgb_path'][0].split('/')
        scene = items[-3]
        
        out_scene_dir = os.path.join(extra_out_dir, scene)
        os.makedirs(out_scene_dir, exist_ok=True)
        
        
        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()
            out = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
            featmaps = out[0]

            ret = render_single_image(ray_sampler=ray_sampler,
                                      ray_batch=ray_batch,
                                      model=model,
                                      projector=projector,
                                      chunk_size=args.chunk_size,
                                      det=True,
                                      N_samples=args.N_samples,
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      enable_rgb=args.enable_rgb,
                                      featmaps=featmaps)

            gt_rgb = data['rgb'][0]
            coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
            coarse_err_map = torch.sum((coarse_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
            coarse_err_map_colored = (colorize_np(coarse_err_map, range=(0., 1.)) * 255).astype(np.uint8)
            coarse_pred_rgb_np = torch.from_numpy(np.clip(coarse_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)).cuda()
            gt_rgb_np = torch.from_numpy(gt_rgb.numpy()[None, ...]).cuda()
            # print(f'coarse_pred_rgb_np.shape: {coarse_pred_rgb_np.shape}')
            # print(f'gt_rgb_np.shape: {gt_rgb_np.shape}')

            coarse_lpips = img2lpips(lpips_loss, gt_rgb_np.permute(0, 3, 1, 2), coarse_pred_rgb_np.permute(0, 3, 1, 2))
            coarse_ssim = img2ssim(gt_rgb_np.permute(0, 3, 1, 2), coarse_pred_rgb_np.permute(0, 3, 1, 2))
            coarse_psnr = img2psnr(gt_rgb_np, coarse_pred_rgb_np)

            # saving outputs ...
            coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)

            gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)

            coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
            coarse_pred_depth_colored = (colorize_np(coarse_pred_depth, range=(0., 1.)) * 255).astype(np.uint8)
            coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()
            coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)

            sum_coarse_psnr += coarse_psnr
            running_mean_coarse_psnr = sum_coarse_psnr / (i + 1)
            sum_coarse_lpips += coarse_lpips
            running_mean_coarse_lpips = sum_coarse_lpips / (i + 1)
            sum_coarse_ssim += coarse_ssim
            running_mean_coarse_ssim = sum_coarse_ssim / (i + 1)

            if ret['outputs_fine'] is not None:
                fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
                fine_pred_rgb_np = torch.from_numpy(np.clip(fine_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)).cuda()

                fine_lpips = img2lpips(lpips_loss, gt_rgb_np.permute(0, 3, 1, 2), fine_pred_rgb_np.permute(0, 3, 1, 2))
                fine_ssim = img2ssim(gt_rgb_np.permute(0, 3, 1, 2), fine_pred_rgb_np.permute(0, 3, 1, 2))
                fine_psnr = img2psnr(gt_rgb_np, fine_pred_rgb_np)

                fine_err_map = torch.sum((fine_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
                fine_err_map_colored = (colorize_np(fine_err_map, range=(0., 1.)) * 255).astype(np.uint8)

                fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}.png'.format(file_id)), fine_pred_rgb)
                fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
                fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                      range=tuple(data['depth_range'].squeeze().cpu().numpy()))
                fine_acc_map = torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()
                fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
            else:
                fine_ssim = fine_lpips = fine_psnr = 0.

            sum_fine_psnr += fine_psnr
            running_mean_fine_psnr = sum_fine_psnr / (i + 1)
            sum_fine_lpips += fine_lpips
            running_mean_fine_lpips = sum_fine_lpips / (i + 1)
            sum_fine_ssim += fine_ssim
            running_mean_fine_ssim = sum_fine_ssim / (i + 1)

            # print("==================\n"
            #       "{}, curr_id: {} \n"
            #       "current coarse psnr: {:03f}, current fine psnr: {:03f} \n"
            #       "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f} \n"
            #       "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
            #       "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n" 
            #       "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
            #       "running mean coarse lpips: {:03f}, running mean fine lpips: {:03f} \n"
            #       "===================\n"
            #       .format(scene_name, file_id,
            #               coarse_psnr, fine_psnr,
            #               running_mean_coarse_psnr, running_mean_fine_psnr,
            #               coarse_ssim, fine_ssim,
            #               running_mean_coarse_ssim, running_mean_fine_ssim,
            #               coarse_lpips, fine_lpips,
            #               running_mean_coarse_lpips, running_mean_fine_lpips
            #               ))

            # [scene_name]
            results_dict[file_id] = {'coarse_psnr': coarse_psnr,
                                                 'fine_psnr': fine_psnr,
                                                 'coarse_ssim': coarse_ssim,
                                                 'fine_ssim': fine_ssim,
                                                 'coarse_lpips': coarse_lpips,
                                                 'fine_lpips': fine_lpips,
                                                 }

    mean_coarse_psnr = sum_coarse_psnr / total_num
    mean_fine_psnr = sum_fine_psnr / total_num
    mean_coarse_lpips = sum_coarse_lpips / total_num
    mean_fine_lpips = sum_fine_lpips / total_num
    mean_coarse_ssim = sum_coarse_ssim / total_num
    mean_fine_ssim = sum_fine_ssim / total_num

    print('------{}-------\n'
          'final coarse psnr: {}, final fine psnr: {}\n'
          'fine coarse ssim: {}, final fine ssim: {} \n'
          'final coarse lpips: {}, fine fine lpips: {} \n'
          .format('total', mean_coarse_psnr, mean_fine_psnr,
                  mean_coarse_ssim, mean_fine_ssim,
                  mean_coarse_lpips, mean_fine_lpips,
                  ))

    # [scene_name]
    results_dict['coarse_mean_psnr'] = mean_coarse_psnr
    results_dict['fine_mean_psnr'] = mean_fine_psnr
    results_dict['coarse_mean_ssim'] = mean_coarse_ssim
    results_dict['fine_mean_ssim'] = mean_fine_ssim
    results_dict['coarse_mean_lpips'] = mean_coarse_lpips
    results_dict['fine_mean_lpips'] = mean_fine_lpips

    f = open("{}/psnr_{}.txt".format(extra_out_dir, model.start_step), "w")
    f.write(str(results_dict))
    f.close()

