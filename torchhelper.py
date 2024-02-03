"""
contains train and evaluate functions
+ a custom class for the isic dataset to allow for loading single classes
"""
import glob
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

def train_model(model, dataloader, num_epochs=3):
    device = 'cuda'
    torch.manual_seed(0)
    iters = len(dataloader)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            outputs = model.predict(x).to(device)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_corrects += torch.sum(outputs.argmax(1) == y)

            if i % 50 == 0:
              print(i, "/", iters)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print('{} loss: {:.4f}, acc: {:.4f}'.format('train',
                                                    epoch_loss,
                                                    epoch_acc))

    return model


def evaluate(adapt_model, dataloader, adapt=False):
  device = 'cuda'
  torch.manual_seed(0)
  total = 0
  correct = 0
  predicted_labels = []
  ground_truth_labels = []

  adapt_model.eval()
  with torch.no_grad():
    for x, y in tqdm(dataloader):
      x = x.to(device)
      y = y.to(device)
      if adapt:
        outputs = adapt_model(x, adapt)
      else:
        outputs = adapt_model(x)
      preds = outputs.argmax(1)
      total += 32
      correct += torch.sum(preds == y)
      predicted_labels.extend(preds.cpu().tolist())
      ground_truth_labels.extend(y.cpu().tolist())

  acc = correct/total
  pr,rc, fscore,_ = precision_recall_fscore_support(ground_truth_labels, predicted_labels, average='binary',zero_division=0)
  print(pr, rc, fscore)
  return acc, pr, rc, fscore

  
class CustomSkinDataset(Dataset):
    def __init__(self, class_name, transform, classes):
        self.imgs_path = "/content/ISIC2019_train/" + str(class_name) + '/'
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            if (class_name in classes):
              for img_path in glob.glob(class_path + "/*.jpg"):
                  self.data.append([img_path, class_name])
        self.class_map = {"ben" : 0, "mel": 1}
        self.img_dim = (224, 224)
        self.transform = transform

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)

        return img, class_id