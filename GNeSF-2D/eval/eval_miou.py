from torch.utils.data import DataLoader
from tqdm import tqdm
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config_parser
from gnesf.sample_ray import RaySamplerSingleImage
from gnesf.render_image import render_single_image
from gnesf.model import IBRNetModel
from utils import *
from gnesf.projection import Projector
from gnesf.data_loaders import dataset_dict

from gnesf.evaluation import SemSegEvaluator


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False
    print(args)

    # Create IBRNet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}/eval'.format(args.rootdir, args.expname)
    print("saving results to {}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    if len(args.eval_scenes) == 1: print('Scene: {}'.format(args.eval_scenes[0]))

    mode = args.mode
    test_dataset = dataset_dict[args.eval_dataset](args, mode, scenes=args.eval_scenes)
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    
    colour_map = test_dataset.colour_map
    
    # # Create evaluator
    dataset_name = 'scannet_sem_seg_val_21cls' if args.num_cls == 21 else 'scannet_sem_seg_val'
    evaluator = SemSegEvaluator(dataset_name=dataset_name, distributed=False, output_dir=extra_out_dir)

    evaluator.reset()
    if args.num_cls == 20: evaluator._ignore_label = -1

    for i, data in enumerate(tqdm(test_loader)):
        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()
            src_rgbs = ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
            src_sems = ray_batch['src_sems'].squeeze(0)
            out = model.feature_net(src_rgbs)
            
            if args.enable_2dsem_seg:
                featmaps, sem_pred2d = out
            else:
                sem_pred2d, featmaps = None, out

            ret = render_single_image(ray_sampler=ray_sampler,
                                    ray_batch=ray_batch,
                                    model=model,
                                    projector=projector,
                                    chunk_size=args.chunk_size,
                                    N_samples=args.N_samples,
                                    inv_uniform=args.inv_uniform,
                                    det=True,
                                    N_importance=args.N_importance,
                                    white_bkgd=args.white_bkgd,
                                    render_stride=args.render_stride,
                                    featmaps=featmaps,
                                    enable_rgb=args.enable_rgb,
                                    enable_semantic=args.enable_semantic,
                                    sem_pred2d=sem_pred2d)


            if ret['outputs_fine'] is not None:
                if args.render_stride != 1:
                    data['sem'] = data['sem'][:, ::args.render_stride, ::args.render_stride]
                evaluator.process_nerf(data, ret['outputs_fine'])
                if args.show:
                    items = data['rgb_path'][0].split('/')
                    scene, file_id = items[-3], items[-1]
                    file_id = file_id.split('.')[0]
                    H, W = ray_sampler.H, ray_sampler.W
                    gt_img = ray_sampler.rgb.reshape(H, W, 3)
                    gt_sem = ray_sampler.sem.reshape(H, W) + 1
                    if args.render_stride != 1:
                        if gt_img is not None:
                            gt_img = gt_img[::args.render_stride, ::args.render_stride]
                        if gt_sem is not None:
                            gt_sem = gt_sem[::args.render_stride, ::args.render_stride]
                    
                    pred_sem = ret['outputs_fine']['sem'].detach().cpu()
                    pred_sem = logits_2_label(pred_sem.view(-1, pred_sem.shape[-1])) + 1
        
                    pred_sem = pred_sem.view([H, W])
                    pred_sem[gt_sem==args.ignore_label] = args.ignore_label + 1
                    pred_sem = pred_sem.view(-1)
                    
                    h, w = gt_sem.shape[:2]
                    rgb_gt = img_HWC2CHW(gt_img)
                    sem_pred = img_HWC2CHW(colour_map[pred_sem].reshape(h, w, 3))
                    sem_gt = img_HWC2CHW(colour_map[gt_sem.reshape(-1)].reshape(h, w, 3))
                    
                    h_max = max(sem_gt.shape[-2], sem_pred.shape[-2], rgb_gt.shape[-2])
                    w_max = max(sem_gt.shape[-1], sem_pred.shape[-1], rgb_gt.shape[-1])
                    
                    sem_im = torch.zeros(3, h_max, 3*w_max)
                    sem_im[:, :sem_gt.shape[-2], :sem_gt.shape[-1]] = rgb_gt*255
                    sem_im[:, :sem_gt.shape[-2], w_max:w_max+sem_pred.shape[-1]] = sem_gt
                    sem_im[:, :sem_pred.shape[-2], 2*w_max:2*w_max+sem_pred.shape[-1]] = sem_pred
                    save_dir = os.path.join(extra_out_dir, scene)
                    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
                    sem_im = img_CHW2HWC(sem_im)
                    sem_im = sem_im.numpy().astype('uint8')
                    cv2.imwrite(os.path.join(save_dir, '{}_demo.jpg'.format(file_id)), np.flip(sem_im, -1))
                    
            else:
                evaluator.process_nerf(data, ret['outputs_coarse'])

    results = evaluator.evaluate()
    print(results)

