from model import SR3
from data import Image_data
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import matplotlib.pyplot as plt

import os


import time


dropout = 0.2
epoch_n = 40
lr = 1e-5
low_size = 64
high_size = 128
root_dir = os.getcwd()
batch_size = 10
load = True
gpu = True
test_only = True
begin_step = 9060000
begin_epoch = 12
print_frec = 2e4
save_frec = 2e4
loss_type = 'l1'
data_dir = '../data'
train_set = '/ffhq'
test_set = '/ffhq'
channel_mults = (1, 2, 4, 8, 8)
res_blocks = 3
inner_channel = 128


if not os.path.exists('./' + str(low_size) + '_' + str(high_size)):
    os.makedirs('./' + str(low_size) + '_' + str(high_size))

save_path = './' + str(low_size) + '_' + str(high_size)+'_3'
train_dir = data_dir + train_set
trainset = Image_data(data_dir=train_dir, low=low_size, high=high_size)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

val_dir = data_dir + test_set
testset = Image_data(data_dir=val_dir, low=low_size, high=high_size, phase='val')
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if gpu else 'cpu')
schedule_opt = {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-6, 'linear_end': 0.01}
model = SR3(
    device=device, img_size=high_size, LR_size=low_size, schedule_opt=schedule_opt, inner_channel=inner_channel,
    loss_type=loss_type, channel_mults=channel_mults, res_blocks=res_blocks, dropout=dropout,
    load=load, load_path='./' + str(low_size) + '_' + str(high_size)+ '_3/model_' + str(begin_step) + '.pt'
)


if not test_only:
    current_step = begin_step
    model.sr3.train()
    i = 0
    for e in range(begin_epoch, epoch_n):
        train_loss = 0
        T1 = time.time()
        print('train epoch: ' + str(e))
        for i, data in enumerate(trainloader):
            train_loss += model.optimize(data) * batch_size
            if current_step % 100 == 0:
                print('step: ' + str(current_step))

            if current_step % save_frec ==0:
                model.save(save_path + '/model_'+str(current_step)+'.pt')

            if current_step % print_frec == 0:
                model.sr3.eval()
                index = random.randint(0, testset.__len__())
                test_imgs = testset.__getitem__(index)
                test_img = torch.unsqueeze(test_imgs["SR"], dim=0).to(device)
                b, c, h, w = test_img.shape
                with torch.no_grad():
                    val_loss = model.sr3.super_resolution(test_img)
                    val_loss = val_loss.sum() / int(b * c * h * w)
                model.sr3.train()

                train_loss = train_loss / len(trainloader)
                print(f'Epoch: {e + 1} / total setp: {current_step} / loss:{train_loss:.3f} / val_loss:{val_loss.item():.3f}')

                # Save example of test images to check training
                plt.figure(figsize=(15, 10))
                plt.subplot(1, 2, 1)
                plt.axis("off")
                plt.title("Low-Resolution Inputs")
                plt.imshow(np.transpose(torchvision.utils.make_grid(test_img,
                                                                    nrow=2, padding=1, normalize=True).cpu(),
                                        (1, 2, 0)))

                plt.subplot(1, 2, 2)
                plt.axis("off")
                plt.title("Super-Resolution Results")
                plt.imshow(np.transpose(torchvision.utils.make_grid(model.test(test_img).detach().cpu(),
                                                                    nrow=2, padding=1, normalize=True), (1, 2, 0)))
                plt.savefig(save_path + '/SuperResolution_' +str(current_step) + '_' + str(index) + '.jpg')
                plt.close()
            current_step += batch_size
else:
    model.sr3.eval()
    index = random.randint(0, testset.__len__())
    test_imgs = testset.__getitem__(index)
    test_img = torch.unsqueeze(test_imgs["SR"], dim=0).to(device)
    b, c, h, w = test_img.shape

    # Save example of test images to check training
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Low-Resolution Inputs")
    plt.imshow(np.transpose(torchvision.utils.make_grid(test_img,
                                                        nrow=2, padding=1, normalize=True).cpu(),
                            (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Super-Resolution Results")
    plt.imshow(np.transpose(torchvision.utils.make_grid(model.test(test_img).detach().cpu(),
                                                        nrow=2, padding=1, normalize=True), (1, 2, 0)))
    plt.savefig(save_path + '/SuperResolution_' + str(index) + '.jpg')
    plt.close()




'''
        model.eval()

        total = 0
        correct = 0
        total_loss = 0

        with torch.no_grad():
            for i, data in enumerate(testloader):
                #image, label = data
                image, label = data['image'], data['label']

                #image = image.unsqueeze(1).to(device)
                image = image.to(device)
                label = label.squeeze(1)
                label = label.long().to(device)

                pred = model(image)
                crop_x = (label.shape[1] - pred.shape[2]) // 2
                crop_y = (label.shape[2] - pred.shape[3]) // 2

                label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

                loss = criterion(pred, label)
                total_loss += loss.item()

                _, pred_labels = torch.max(pred, dim=1)

                total += label.shape[0] * label.shape[1] * label.shape[2]
                correct += (pred_labels == label).sum().item()

            acc = correct / total
            loss_val = total_loss / testset.__len__()
            print('Accuracy: %.4f ---- Loss: %.4f' % (acc, loss_val))
            all_acc.append(acc)
            all_total.append(loss_val)
        wandb.log({"train loss": loss_train, "accuracy": acc, "test loss": loss_val})
        wandb.watch(model)

    print('total time: %s s' % total_time)
    plt.plot(axis_x, all_loss, linestyle='-', color='steelblue', label='train loss')
    plt.plot(axis_x, all_acc, linestyle='-', color='indianred', label='accuracy')
    plt.plot(axis_x, all_total, linestyle='-', color='seagreen', label='test loss')
    plt.xlim((0, epoch_n + 1))
    plt.ylim((0, 1))
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('./'+ str(image_size) + '/result.png')
    plt.show()


#testing and visualization

model.eval()

output_masks = []
output_labels = []

total = 0
correct = 0
with torch.no_grad():
    for i in range(testset.__len__()):
        image, labels = testset.__getitem__(i)['image'], testset.__getitem__(i)['label']

        input_image = image.unsqueeze(0).to(device)
        labels = labels.squeeze(0)
        pred = model(input_image)

        output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()

        crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
        crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
        labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].numpy()

        output_masks.append(output_mask)
        output_labels.append(labels)

        total += labels.shape[0] * labels.shape[1]
        correct += (output_mask == labels).sum().item()

acc = correct / total
print('Accuracy: %.4f' % (acc))
fig, axes = plt.subplots(testset.__len__(), 2, figsize = (20, 20))

for i in range(testset.__len__()):
  axes[i, 0].imshow(output_labels[i])
  axes[i, 0].axis('off')
  axes[i, 1].imshow(output_masks[i])
  axes[i, 1].axis('off')
#plt.savefig('./'+ str(image_size) + '/test.png')
plt.show()
'''
