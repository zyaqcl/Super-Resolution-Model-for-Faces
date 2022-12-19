import torch
import torch.nn as nn
import math
import numpy as np
from functools import partial


class PositionalEncoding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, noise_level):
    count = self.dim // 2
    step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
    encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
    encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
    return encoding


class FeatureWiseAffine(nn.Module):
  def __init__(self, in_channels, out_channels, use_affine_level=False):
    super(FeatureWiseAffine, self).__init__()
    self.use_affine_level = use_affine_level
    self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

  def forward(self, x, noise_embed):
    noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
    if self.use_affine_level:
      gamma, beta = noise.chunk(2, dim=1)
      x = (1 + gamma) * x + beta
    else:
      x = x + noise
    return x


class Swish(nn.Module):
  def forward(self, x):
    return x * torch.sigmoid(x)


class downStep(nn.Module):
  def __init__(self, dim):
    super(downStep, self).__init__()
    self.block = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

  def forward(self, x):
    return self.block(x)

class upStep(nn.Module):
  def __init__(self, dim):
    super(upStep, self).__init__()
    self.up = nn.Upsample(scale_factor=2, mode='nearest')
    self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.up(x)
    return self.conv(x)

class Block(nn.Module):
  def __init__(self, dim, dim_out, groups, dropout=0):
    super(Block, self).__init__()
    self.block = nn.Sequential\
    (
        nn.GroupNorm(groups, dim),
        Swish(),
        nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
        nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
    )

  def forward(self, x):
    return self.block(x)

class SelfAttention(nn.Module):
  def __init__(self, dim, n_head, norm_groups=32):
    super().__init__()
    self.n_head = n_head
    self.norm = nn.GroupNorm(norm_groups, dim)
    self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
    self.out = nn.Conv2d(dim, dim, kernel_size=1)

  def forward(self, input):
    batch, channel, height, width = input.shape
    n_head = self.n_head
    head_dim = channel // n_head

    norm = self.norm(input)
    qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
    query, key, value = qkv.chunk(3, dim=2)

    attn = torch.einsum(
        "bnchw, bncyx -> bnhwyx", query, key
      ).contiguous() / math.sqrt(channel)
    attn = attn.view(batch, n_head, height, width, -1)
    attn = torch.softmax(attn, -1)
    attn = attn.view(batch, n_head, height, width, height, width)

    out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
    out = self.out(out.view(batch, channel, height, width))

    return out + input


class ResBlock(nn.Module):
  def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
               num_heads=1, use_affine_level=False, norm_groups=32, att=False):
    super().__init__()
    self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
    self.block1 = Block(dim, dim_out, groups=norm_groups)
    self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
    self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    self.att = att
    if att:
      self.attn = SelfAttention(dim_out, n_head=num_heads, norm_groups=norm_groups)

  def forward(self, x, time_emb):
    y = self.block1(x)
    y = self.noise_func(y, time_emb)
    y = self.block2(y)
    x = y + self.res_conv(x)
    if self.att:
      x = self.attn(x)
    return x

class UNet(nn.Module):
  def __init__(self, in_channel=6, out_channel=3, inner_channel=32, channel_mults=[1, 2, 4, 8, 8],
               norm_groups=32, res_blocks=3, attn_res=[16], dropout=0, img_size=128):
    super(UNet, self).__init__()

    noise_level_channel = inner_channel
    self.noise_level_mlp = nn.Sequential(
      PositionalEncoding(inner_channel),
      nn.Linear(inner_channel, inner_channel * 4),
      Swish(),
      nn.Linear(inner_channel * 4, inner_channel)
    )

    num_mults = len(channel_mults)
    pre_channel = inner_channel
    feat_channels = [pre_channel]
    now_res = img_size

    downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
    for ind in range(num_mults):
      is_last = (ind == num_mults - 1)
      use_attn = (now_res in attn_res)
      channel_mult = inner_channel * channel_mults[ind]
      for _ in range(0, res_blocks):
        downs.append(ResBlock(
          pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
          norm_groups=norm_groups, dropout=dropout, att=use_attn))
        feat_channels.append(channel_mult)
        pre_channel = channel_mult
      if not is_last:
        downs.append(downStep(pre_channel))
        feat_channels.append(pre_channel)
        now_res = now_res // 2
    self.downs = nn.ModuleList(downs)

    self.mid = nn.ModuleList([
      ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
               norm_groups=norm_groups, dropout=dropout, att=True),
      ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
               norm_groups=norm_groups, dropout=dropout, att=False)
    ])

    ups = []
    for ind in reversed(range(num_mults)):
      is_last = (ind < 1)
      use_attn = (now_res in attn_res)
      channel_mult = inner_channel * channel_mults[ind]
      for _ in range(0, res_blocks + 1):
        ups.append(ResBlock(
          pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
          norm_groups=norm_groups, dropout=dropout, att=use_attn))
        pre_channel = channel_mult
      if not is_last:
        ups.append(upStep(pre_channel))
        now_res = now_res * 2

    self.ups = nn.ModuleList(ups)

    self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

  def forward(self, x, time):
    t = self.noise_level_mlp(time)
    feats = []
    for layer in self.downs:
      if isinstance(layer, ResBlock):
        x = layer(x, t)
      else:
        x = layer(x)
      feats.append(x)

    for layer in self.mid:
      x = layer(x, t)

    for layer in self.ups:
      if isinstance(layer, ResBlock):
        x = layer(torch.cat((x, feats.pop()), dim=1), t)
      else:
        x = layer(x)

    return self.final_conv(x)

class Diffusion(nn.Module):
  def __init__(self, model, device, img_size, LR_size, channels=3):
    super().__init__()
    self.channels = channels
    self.model = model.to(device)
    self.img_size = img_size
    self.LR_size = LR_size
    self.device = device

  def set_loss(self, loss_type):
    if loss_type == 'l1':
      self.loss_func = nn.L1Loss(reduction='sum')
    elif loss_type == 'l2':
      self.loss_func = nn.MSELoss(reduction='sum')
    else:
      raise NotImplementedError()

  def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
    if schedule == 'linear':
      betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'quad':
      betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    return betas

  def set_new_noise_schedule(self, schedule_opt):
    to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

    betas = self.make_beta_schedule(
      schedule=schedule_opt['schedule'],
      n_timestep=schedule_opt['n_timestep'],
      linear_start=schedule_opt['linear_start'],
      linear_end=schedule_opt['linear_end'])
    betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
    self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

    self.num_timesteps = int(len(betas))
      # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
    self.register_buffer('betas', to_torch(betas))
    self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
    self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
    self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
    self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

      # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
    variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    self.register_buffer('variance', to_torch(variance))
      # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
    self.register_buffer('posterior_mean_coef1',
                           to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
    self.register_buffer('posterior_mean_coef2',
                           to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

  # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
  def predict_start(self, x_t, t, noise):
    return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
  def q_posterior(self, x_start, x_t, t):
    posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
    posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
    return posterior_mean, posterior_log_variance_clipped




  def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
    batch_size = x.shape[0]
    noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
    if condition_x is not None:
      x_recon = self.predict_start(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))
    else:
      x_recon = self.predict_start(x, t, noise=self.model(x, noise_level))

    if clip_denoised:
      x_recon.clamp_(-1., 1.)

    mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
    return mean, posterior_log_variance



  @torch.no_grad()
  def p_sample(self, x, t, clip_denoised=True, condition_x=None):
    t = int(t)
    mean, log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
    noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
    return mean + noise * (0.5 * log_variance).exp()


  @torch.no_grad()
  def super_resolution(self, x_in):
    img = torch.randn_like(x_in, device=x_in.device)
    for i in reversed(range(0, self.num_timesteps)):
      img = self.p_sample(img, i, condition_x=x_in)
    return img


  def p_losses(self, x_in, noise=None):
    x_start = x_in['HR']
    [b, c, h, w] = x_start.shape
    t = np.random.randint(1, self.num_timesteps + 1)
    continuous_sqrt_alpha_cumprod = torch.FloatTensor(
      np.random.uniform(self.sqrt_alphas_cumprod_prev[t - 1],
                        self.sqrt_alphas_cumprod_prev[t], size=b)
    ).to(x_start.device)
    continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
      -1, 1, 1, 1)

    noise = torch.randn_like(x_start).to(x_start.device)
    x_noisy = continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise


    x_recon = self.model(torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

    loss = self.loss_func(noise, x_recon)
    return loss

  def forward(self, x, *args, **kwargs):
    return self.p_losses(x, *args, **kwargs)



class SR3():
  def __init__(self, device, img_size, LR_size, loss_type, schedule_opt, load_path=None,
               load=False, in_channel=6, out_channel=3, inner_channel=32, norm_groups=8,
               channel_mults=(1, 2, 4, 8, 8), res_blocks=2, dropout=0, lr=1e-5, distributed=False):
    super(SR3, self).__init__()
    self.device = device
    self.img_size = img_size
    self.LR_size = LR_size

    model = UNet(in_channel, out_channel, inner_channel, channel_mults, norm_groups, res_blocks, dropout=dropout, img_size=img_size)
    self.sr3 = Diffusion(model, device, img_size, LR_size, out_channel)

    # Apply weight initialization & set loss & set noise schedule
    self.sr3.apply(self.weights_init_orthogonal)
    self.sr3.set_loss(loss_type)
    self.sr3.set_new_noise_schedule(schedule_opt)

    if distributed:
      assert torch.cuda.is_available()
      self.sr3 = nn.DataParallel(self.sr3)

    self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

    params = sum(p.numel() for p in self.sr3.parameters())
    print(f"Number of model parameters : {params}")

    if load:
      print('loading model')
      self.load(load_path)

  def weights_init_orthogonal(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
      nn.init.orthogonal_(m.weight.data, gain=1)
      if m.bias is not None:
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
      nn.init.orthogonal_(m.weight.data, gain=1)
      if m.bias is not None:
        m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
      nn.init.constant_(m.weight.data, 1.0)
      nn.init.constant_(m.bias.data, 0.0)


  def optimize(self, data):
    for k, i in data.items():
      data[k] = i.to(self.device)
    self.data = data
    self.optimizer.zero_grad()
    l = self.sr3(self.data)
    b, c, h, w = self.data['HR'].shape
    l = l.sum() / int(b * c * h * w)
    l.backward()
    self.optimizer.step()
    return l.item()


  def test(self, imgs):
    self.sr3.eval()
    with torch.no_grad():
      if isinstance(self.sr3, nn.DataParallel):
        result_SR = self.sr3.module.super_resolution(imgs)
      else:
        result_SR = self.sr3.super_resolution(imgs)
    self.sr3.train()
    return result_SR

  def save(self, save_path):
    network = self.sr3.model
    if isinstance(self.sr3, nn.DataParallel):
      network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
      state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

  def load(self, load_path):
    network = self.sr3.model
    #network = self.sr3
    if isinstance(self.sr3, nn.DataParallel):
      network = network.module
    network.load_state_dict(torch.load(load_path))
    print("Model loaded successfully")