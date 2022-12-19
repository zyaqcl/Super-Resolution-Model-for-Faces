import torch

from model import SR3
import os
import numpy as np
from data import Image_data
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import torchvision


class Sampler:
    def __init__(self, Diffusion, device, data_predict=False):
        alphas = Diffusion.alphas_cumprod_prev
        betas = Diffusion.betas
        log_alphas = torch.log(alphas)
        log_alphas = torch.log(1 - betas).cumsum(dim=0)
        # It turns out in the experiment that interpolating on log_alphas here can acquire higher accuracy than interpolating on log of alphas.
        self.alphas = alphas.to(device)
        self.log_alpha = log_alphas.to(device)
        self.total = len(alphas)
        self.model = Diffusion
        self.t = torch.linspace(0., 1., self.total + 1)[1:].to(device)
        self.t_T = 1.
        self.t_0 = 1 / self.total
        self.device = device
        self.data_predict = data_predict

    def sample(self, x, order, steps):
        show = torchvision.transforms.ToPILImage()
        if order == 3:
            k = steps // 3
            if steps % 3 == 0:
                orders = [3, ] * (k - 1) + [2, 1]
            elif steps % 3 == 1:
                orders = [3, ] * k + [1]
            else:
                orders = [3, ] * k + [2]
        elif order == 2:
            k = steps // 2
            orders = [2, ] * k
            if steps % 2 != 0:
                orders = orders + [1]
        else:
            orders = [1, ] * steps
        time_step = torch.linspace(self.t_T, self.t_0, steps + 1)[torch.cumsum(torch.tensor(orders), dim=0)]
        time_step = torch.cat([torch.tensor([1., ]), time_step]).to(self.device)
        img = torch.randn_like(x, device=x.device) * 0.5 + 0.5
        for step, order in enumerate(orders):
            s, t = time_step[step], time_step[step + 1]
            s, t = s.unsqueeze(0), t.unsqueeze(0)
            lambda_s = self.forward_lambda(s)
            lambda_t = self.forward_lambda(t)
            h = lambda_t - lambda_s
            if order == 3:
                r1 = 1 / 3
                r2 = 2 / 3
                lambda_s1 = lambda_s + r1 * h
                lambda_s2 = lambda_s + r2 * h
                s1 = self.inverse_lambda(lambda_s1)
                s2 = self.inverse_lambda(lambda_s2)
                if not self.data_predict:
                    # i1 = self.model.p_sample(img, s.item() * self.total - 1, condition_x=x)
                    i1 = self.noise(img, (s.item() - 1 / self.total) * 1000, x)
                    alpha_t, alpha_s, alpha_s1, alpha_s2 = self.alpha_from_t(t), self.alpha_from_t(
                        s), self.alpha_from_t(s1), self.alpha_from_t(s2)
                    sigma_t, sigma_s, sigma_s1, sigma_s2 = self.sigma_from_alpha(alpha_t), self.sigma_from_alpha(
                        alpha_s), self.sigma_from_alpha(alpha_s1), self.sigma_from_alpha(alpha_s2)
                    p1 = torch.expm1(r1 * h)
                    p2 = torch.expm1(r2 * h)
                    u1 = torch.sqrt(alpha_s1 / alpha_s) * img - sigma_s1 * p1 * i1
                    # i2 = self.model.p_sample(u1, s1.item() * self.total - 1, condition_x=x)
                    i2 = self.noise(u1, (s1.item() - 1 / self.total) * 1000, x)
                    d1 = i2 - i1
                    u2 = torch.sqrt(alpha_s2 / alpha_s) * img - sigma_s2 * p2 * i1 - sigma_s2 * r2 / r1 * (
                                p2 / (r2 * h) - 1) * d1
                    # i3 = self.model.p_sample(u2, s2.item() * self.total - 1, condition_x=x)
                    i3 = self.noise(u2, (s1.item() - 1 / self.total) * 1000, x)
                    d2 = i3 - i1
                    p3 = torch.expm1(h)
                    img = torch.sqrt(alpha_t / alpha_s) * img - sigma_t * p3 * i1 - sigma_t / r2 * (p3 / h - 1) * d2
                else:
                    alpha_t, alpha_s, alpha_s1, alpha_s2 = self.alpha_from_t(t), self.alpha_from_t(
                        s), self.alpha_from_t(s1), self.alpha_from_t(s2)
                    sigma_t, sigma_s, sigma_s1, sigma_s2 = self.sigma_from_alpha(alpha_t), self.sigma_from_alpha(
                        alpha_s), self.sigma_from_alpha(alpha_s1), self.sigma_from_alpha(alpha_s2)
                    alpha_t, alpha_s, alpha_s1, alpha_s2 = torch.sqrt(alpha_t), torch.sqrt(alpha_s), torch.sqrt(
                        alpha_s1), torch.sqrt(alpha_s2)
                    p1 = torch.expm1(-r1 * h)
                    p2 = torch.expm1(-r2 * h)
                    p3 = torch.expm1(-h)
                    i1 = self.data(img, (s.item() - 1 / self.total) * 1000, x)
                    u1 = (sigma_s1 / sigma_s) * img - (alpha_s1 * p1) * i1
                    i2 = self.data(u1, (s1.item() - 1 / self.total) * 1000, x)
                    d1 = i2 - i1
                    u2 = sigma_s2 / sigma_s * img - alpha_s2 * p2 * i1 + r2 / r1 * alpha_s2 * (p2 / (r2 * h) + 1) * d1
                    i3 = self.data(u2, (s2.item() - 1 / self.total) * 1000, x)
                    d2 = i3 - i1
                    img = (sigma_t / sigma_s) * img - (alpha_t * p3) * i1 + (1. / r2) * (alpha_t * (p3 / h + 1)) * d2

            elif order == 2:
                r1 = 0.5
                lambda_s1 = lambda_s + r1 * h
                s1 = self.inverse_lambda(lambda_s1)
                if not self.data_predict:

                    # i1 = self.model.p_sample(img, s.item() * self.total - 1, condition_x=x)
                    i1 = self.noise(img, (s.item() - 1 / self.total) * 1000, x)
                    alpha_t, alpha_s, alpha_s1 = self.alpha_from_t(t), self.alpha_from_t(s), self.alpha_from_t(s1)
                    sigma_t, sigma_s, sigma_s1 = self.sigma_from_alpha(alpha_t), self.sigma_from_alpha(
                        alpha_s), self.sigma_from_alpha(alpha_s1)
                    p1 = torch.expm1(r1 * h)
                    u = torch.sqrt(alpha_s1 / alpha_s) * img - sigma_s1 * p1 * i1
                    # i2 = self.model.p_sample(u, s1.item() * self.total - 1, condition_x=x)
                    i2 = self.noise(u, (s1.item() - 1 / self.total) * 1000, x)
                    d1 = i2 - i1
                    p2 = torch.expm1(h)
                    img = torch.sqrt(alpha_t / alpha_s) * img - sigma_t * p2 * i1 - sigma_t / 2 / r1 * p2 * d1


                else:
                    alpha_t, alpha_s, alpha_s1 = self.alpha_from_t(t), self.alpha_from_t(
                        s), self.alpha_from_t(s1)
                    sigma_t, sigma_s, sigma_s1 = self.sigma_from_alpha(alpha_t), self.sigma_from_alpha(
                        alpha_s), self.sigma_from_alpha(alpha_s1)
                    alpha_t, alpha_s, alpha_s1 = torch.sqrt(alpha_t), torch.sqrt(alpha_s), torch.sqrt(alpha_s1)

                    p1 = torch.expm1(-r1 * h)
                    p2 = torch.expm1(-h)

                    i1 = self.data(img, (s.item() - 1 / self.total) * 1000, x)
                    u1 = sigma_s1 / sigma_s * img - (alpha_s1 * p1) * i1
                    i2 = self.data(u1, (s1.item() - 1 / self.total) * 1000, x)
                    d1 = i2 - i1
                    img = sigma_t / sigma_s * img - alpha_t * p2 * i1 - 0.5 / r1 * alpha_t * p2 * d1

            else:
                alpha_t, alpha_s = self.alpha_from_t(t), self.alpha_from_t(s)
                sigma_t, sigma_s = self.sigma_from_alpha(alpha_t), self.sigma_from_alpha(alpha_s)
                if not self.data_predict:

                    i1 = self.noise(img, (s.item() - 1 / self.total) * 1000, x)
                    p1 = torch.expm1(h)
                    img = torch.sqrt(alpha_t / alpha_s) * img - sigma_t * p1 * i1


                else:
                    p1 = torch.expm1(-h)
                    i1 = self.data(img, (s.item() - 1 / self.total) * 1000, x)
                    img = sigma_t / sigma_s * img - torch.sqrt(alpha_t) * p1 * i1
            show(img.cpu().squeeze(0)).show()
        return img

    def data(self, x, t, condition_x):
        noise = self.noise(x, t, condition_x)
        t = int(t)
        x0 = self.model.predict_start(x, t, noise=noise)
        return x0

    @torch.no_grad()
    def noise(self, x, t, condition_x):
        t = int(t)
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.model.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(
            self.device)
        noise = self.model.model(torch.cat([condition_x, x], dim=1), noise_level)
        return noise

    def forward_lambda(self, t):
        alpha = self.interpolate(t)
        sigma = 0.5 * torch.log(1 - torch.exp(alpha))
        return 0.5 * alpha - sigma

    def inverse_lambda(self, lam):
        log_alpha = -0.5 * torch.logaddexp(torch.zeros(1).to(self.device), -2. * lam)
        t = self.interpolate(log_alpha, type='lambda')
        return t

    def alpha_from_t(self, t):
        alpha = self.interpolate(t)
        return torch.exp(alpha)

    def sigma_from_alpha(self, alpha):
        sigma = torch.sqrt(1 - alpha)
        return sigma

    def interpolate(self, x, type='alpha'):
        if type == 'lambda':
            xs = torch.flip(0.5 * self.log_alpha, [0])
            ys = torch.flip(self.t, [0])
        else:
            xs = self.t
            ys = self.log_alpha
        l = torch.tensor(self.total).unsqueeze(0).to(self.device)
        all_x = torch.cat([x, xs])
        sorted_t, indices = torch.sort(all_x)
        location = torch.argmin(indices).unsqueeze(0)
        start_location = location - 1
        start_location_x = start_location if location != l else l - 2
        start_location_x = start_location_x if location != torch.tensor([0., ]).to(self.device) else torch.tensor([1., ]).to(
            self.device)
        end_location_x = start_location_x + 2 if start_location_x == start_location else start_location_x + 1
        start_x = sorted_t[start_location_x]
        end_x = sorted_t[end_location_x]
        start_location_y = start_location if location != l else l - 2
        start_location_y = start_location_y if location != torch.tensor([0., ]).to(self.device) else torch.tensor([0., ]).to(
            self.device)
        start_y = ys[start_location_y]
        end_y = ys[start_location_y + 1]
        result = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
        return result



if __name__ == '__main__':
    dropout = 0.2
    epoch_n = 20
    lr = 1e-4
    low_size = 64
    high_size = 128
    root_dir = os.getcwd()
    batch_size = 10
    load = True
    gpu = True
    test_only = False
    begin_step = 1280000
    begin_epoch = 0
    print_frec = 2e4
    save_frec = 2e4
    loss_type = 'l1'
    data_dir = '../data'
    train_set = '/ffhq'
    test_set = '/ffhq'
    channel_mults = (1, 2, 4, 8, 8)
    res_blocks = 3
    inner_channel = 32
    save_path = './' + str(low_size) + '_' + str(high_size)
    device = torch.device('cuda' if gpu else 'cpu')
    schedule_opt = {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-6, 'linear_end': 0.01}
    model = SR3(
        device=device, img_size=high_size, LR_size=low_size, schedule_opt=schedule_opt, inner_channel=inner_channel,
        loss_type=loss_type, channel_mults=channel_mults, res_blocks=res_blocks, dropout=dropout,
        load=load, load_path='./' + str(low_size) + '_' + str(high_size) + '/model_' + str(begin_step) + '.pt'
    )
    sampler = Sampler(model.sr3, device)
    val_dir = data_dir + test_set
    testset = Image_data(data_dir=val_dir, low=low_size, high=high_size, phase='val')
    testloader = DataLoader(testset, batch_size=batch_size)
    for i in range(1):
        index = random.randint(0, testset.__len__())
        test_imgs = testset.__getitem__(index)
        test_img = torch.unsqueeze(test_imgs["SR"], dim=0).to(device)
        x_sample = sampler.sample(test_img, order=3, steps=20)
        torchvision.utils.save_image(torchvision.utils.make_grid(x_sample.detach().cpu(), normalize=True).cpu(),
                                     './sample/' + str(index) + '_3_20.jpg')
        '''
        plt.figure()
        plt.imshow(np.transpose(torchvision.utils.make_grid(x_sample,
                                                            nrow=2, padding=1, normalize=True).cpu(),
                                (1, 2, 0)))
        plt.savefig('./' + str(low_size) + '_' + str(high_size) + '/' + str(index) + '_50.jpg')
        plt.close()

        x_test = model.test(test_img)
        plt.figure()
        plt.imshow(np.transpose(torchvision.utils.make_grid(x_test,
                                                            nrow=2, padding=1, normalize=True).cpu(),
                                (1, 2, 0)))
        plt.savefig('./' + str(low_size) + '_' + str(high_size) + '/' + str(index) + '_2000.jpg')
        plt.close()
        '''
