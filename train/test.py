import torch
import torch.optim as optim
import torch.nn.functional as F
from models.doc_encoder import DocEncoder
from batchers.wiki_s2v_batch import WikiS2vBatch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from configs import configs
from tqdm import tqdm
import pickle

print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, tr_mask, in_mask, out_mask) in enumerate(test_loader):
            data, tr_mask = data.to(device), tr_mask.to(device)
            in_mask, out_mask = in_mask.to(device), out_mask.to(device)

            output = model(data*in_mask, mask=tr_mask)

            # dataset_test.check_accuracy()
            test_loss += F.mse_loss(output*out_mask, data*out_mask, reduction='mean').detach()

    test_loss /= batch_idx+1
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    return test_loss

def check_model(model, device, dataset_test):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sent_idx in range(100):
            art_idx = 10
            test_sent_idx = sent_idx
            data, target, tr_mask, in_mask, out_mask, test_s2v, test_label = dataset_test.get_test_data(art_idx=art_idx, start_sent_idx=15,
                                                                                                        test_sent_idx=test_sent_idx, init=(sent_idx == 0))
            data, target, tr_mask = data.to(device), target.to(device), tr_mask.to(device)
            in_mask, out_mask = in_mask.to(device), out_mask.to(device)
            test_s2v, test_label = test_s2v.to(device), test_label.to(device)

            data = torch.nn.functional.normalize(data, dim=2)
            target = torch.nn.functional.normalize(target, dim=2)
            test_s2v = torch.nn.functional.normalize(test_s2v, dim=2)

            # data, target, tr_mask = data.to(device), target.to(device), tr_mask.to(device)
            # in_mask, out_mask = in_mask.to(device), out_mask.to(device)
            # output = model(data*in_mask, mask=tr_mask)
            output, pred_class = model(data, test_s2v, mask=tr_mask)
            pred_class = torch.sigmoid(pred_class)
            print("\r"+str(int(float(pred_class[0, -1, 1])*100))+"%")
            if pred_class[0, -1, 1] < 0.5:
                correct += 1
            total += 1
            # print(correct/float(total))

        # out_s2v = output[0, test_sent_idx, :]
        # print(out_s2v)
        # dataset_test.check_accuracy(out_s2v)
        # loss = (F.mse_loss(output*out_mask, data*out_mask, reduction='sum')/torch.sum(out_mask)).detach()
        # print("loss: "+str(loss))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = DocEncoder(config)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
print(model)
checkpoint = torch.load('./train/save/'+config['name'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
del checkpoint

dataset_train = WikiS2vBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=4)

dataset_test = WikiS2vBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1,
    shuffle=False, num_workers=0)

check_model(model, device, dataset_test)
# for epoch in range(1, 2):
#     current_test_loss = test(model, device, data_loader_test, epoch)
