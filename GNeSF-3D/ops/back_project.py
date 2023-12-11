import torch
from torch.nn.functional import grid_sample
import torch.nn.functional as F


def back_project(coords, origin, voxel_size, feats, KRcam):
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume
    local

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    '''
    n_views, bs, c, h, w = feats.shape

    feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda() # (num of voxels, c + 1)
    count = torch.zeros(coords.shape[0]).cuda()

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        grid_batch = coords_batch * voxel_size + origin_batch.float() # (num of voxels, 3) 3D global coord
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1) # (n_views, num of voxels, 3)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous() # (n_views, 3, num of voxels)
        nV = rs_grid.shape[-1] # n_views
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1) # (n_views, 4, num of voxels)

        # Project grid
        im_p = proj_batch @ rs_grid # 3D coord -> 2D
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2] # (n_views, num of voxels)
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1) # ???  # (n_views, num of voxels, 2)
        mask = im_grid.abs() <= 1  # (n_views, num of voxels) inside bound; True: valid
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2) # (n_views, 1, num of voxels, 2)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True) # (n_views, c, 1, num of voxels)

        features = features.view(n_views, c, -1) # (n_views, c, num of voxels)
        mask = mask.view(n_views, -1)  # (n_views, num of voxels)
        im_z = im_z.view(n_views, -1)  # (n_views, num of voxels)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        count[batch_ind] = mask.sum(dim=0).float() # number of valid views

        # aggregate multi view
        features = features.sum(dim=0) # sum features from different views (c, num of voxels)
        mask = mask.sum(dim=0) # (num of voxels)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1 # x feature / 0
        in_scope_mask = mask.unsqueeze(0) # number of valid views
        features /= in_scope_mask # average
        features = features.permute(1, 0).contiguous() # (c, num of voxels)

        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous() # (num of voxels, 1)
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1) # (num of voxels, c+1)

        feature_volume_all[batch_ind] = features
    return feature_volume_all, count # [number of voxel x bs, c+1]


def back_project_views(coords, origin, voxel_size, feats, KRcam):
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume
    local

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    '''
    n_views, bs, c, h, w = feats.shape

    feature_volume_all = torch.zeros(coords.shape[0], n_views, c + 1).cuda() # (num of voxels, n_views, c + 1)
    count = torch.zeros(coords.shape[0]).cuda()

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        grid_batch = coords_batch * voxel_size + origin_batch.float() # (num of voxels, 3) 3D global coord
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1) # (n_views, num of voxels, 3)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous() # (n_views, 3, num of voxels)
        nV = rs_grid.shape[-1] # n_views
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1) # (n_views, 4, num of voxels)

        # Project grid
        im_p = proj_batch @ rs_grid # 3D coord -> 2D
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2] # (n_views, num of voxels)
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1) # ???  # (n_views, num of voxels, 2)
        mask = im_grid.abs() <= 1  # (n_views, num of voxels) inside bound; True: valid
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2) # (n_views, 1, num of voxels, 2)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True) # (n_views, c, 1, num of voxels)

        features = features.view(n_views, c, -1) # (n_views, c, num of voxels)
        mask = mask.view(n_views, -1)  # (n_views, num of voxels)
        im_z = im_z.view(n_views, -1)  # (n_views, num of voxels)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0
        
        features = features.permute(2, 0, 1).contiguous() # (num of voxels, n_views, c)
        im_z = im_z.permute(1, 0).contiguous()[..., None] # (num of voxels, n_views)

        count[batch_ind] = mask.sum(dim=0).float() # number of valid views
        
        features = torch.cat([features, im_z], dim=-1) # (num of voxels, n_views, c+1)

        feature_volume_all[batch_ind] = features
    return feature_volume_all, count # [number of voxel x bs, n_views, c+1]



def normolize_pts(x, voxel_dim, voxel_size):
    nxyz = voxel_dim * voxel_size
    
    x = 2 * (x / nxyz) - 1 # normalize to [-1, 1] b, n_rays, N_samples, 3
    x = x.flip(-1)
    
    return x[None]


@torch.no_grad()
def is_occlusion(origin, coords, volume_sdf, positions, voxel_size, N_sample=10):
    final_pts =  coords.permute(2, 0, 1)    # (num of voxels, n_views, 3) [mask.any(0)]
    o = positions[None]
    
    diff = final_pts - o
    dists = torch.norm(diff, dim=-1, keepdim=True) # 
    dirs = diff / dists
    nears = 1
    fars = dists
    
    _t = torch.linspace(0, 1, N_sample).float().to(o.device)
    d = nears * (1 - _t) + fars * _t                             # n_vol, n_views, N_samples
    pts = o[..., None, :] + dirs[..., None, :] * d[..., :, None]
    pts -= origin[None, None] # shift to local coordinate
    
    voxel_dim = torch.tensor(volume_sdf.shape[:3]).to(o.device)
    pts = normolize_pts(pts, voxel_dim, voxel_size)         # 1, n_vol, n_views, N_samples, 3
    volume_sdf = volume_sdf[None].permute(0, 4, 1, 2, 3)    # 1, 1, z, y, x
    sdf = F.grid_sample(volume_sdf, pts, mode='bilinear', padding_mode='border', align_corners = True) # ????  b, c, n_pts, 1, 1
    sdf = sdf.squeeze(0).squeeze(0) # .squeeze(1)
    
    # occlusion
    signs = sdf[..., 1:] * sdf[..., :-1]
    mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
    valid1 = mask.sum(-1) < 2
    
    valid = valid1.permute(1, 0)
    
    return valid


def back_project_batch(coords, origin, voxel_size, feats, KRcam, volume_sdf=None, positions=None):
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume
    local

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    '''
    n_views, c, h, w = feats.shape

    coords_batch = coords

    coords_batch = coords_batch.view(-1, 3)
    origin_batch = origin.unsqueeze(0)
    feats_batch = feats
    proj_batch = KRcam

    grid_batch = coords_batch * voxel_size + origin_batch.float() # (num of voxels, 3) 3D global coord
    rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1) # (n_views, num of voxels, 3)
    rs_grid = rs_grid.permute(0, 2, 1).contiguous() # (n_views, 3, num of voxels)
    nV = rs_grid.shape[-1] # num of voxels
    rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1) # (n_views, 4, num of voxels)
    

    # Project grid
    im_p = proj_batch @ rs_grid # 3D coord -> 2D
    im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2] # (n_views, num of voxels)
    im_x = im_x / im_z
    im_y = im_y / im_z

    im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1) # ???  # (n_views, num of voxels, 2)
    mask = im_grid.abs() <= 1  # (n_views, num of voxels) inside bound; True: valid
    mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

    visible = is_occlusion(origin_batch, rs_grid[:, :3], volume_sdf, positions, voxel_size)
    mask = mask & visible
    
    feats_batch = feats_batch.view(n_views, c, h, w)
    im_grid = im_grid.view(n_views, 1, -1, 2) # (n_views, 1, num of voxels, 2)
    features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True) # (n_views, c, 1, num of voxels)

    features = features.view(n_views, c, -1) # (n_views, c, num of voxels)
    mask = mask.view(n_views, -1)  # (n_views, num of voxels)
    im_z = im_z.view(n_views, -1)  # (n_views, num of voxels)
    # remove nan
    features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
    im_z[mask == False] = 0
    
    features = features.permute(2, 0, 1).contiguous() # (num of voxels, n_views, c)
    im_z = im_z.permute(1, 0).contiguous()[..., None] # (num of voxels, n_views)

    mask = mask.permute(1, 0).contiguous() # (num of voxels, n_views)

    features = torch.cat([features, im_z], dim=-1) # (num of voxels, n_views, c+1)

    return features, mask
