import torch, os, sys, cv2, json, argparse, random, csv, math
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func

import numpy as np 

from torchvision import transforms, models
from tensorboardX import SummaryWriter


class ChairShapenetDataset(Dataset):
	"""
	Args:
		csv_file (string): path to the csv file containing the names of the images in imagenet
		root_dir (string): Directory with all the images
		transform (callable, optional): Optional transform to be applied on a sample
	"""

	def __init__(self,csv_file, root_dir):
		self.poses = []
		f = csv.reader(open(csv_file), delimiter=',')
		for item in f:
			self.poses.append(item)

		self.root_dir = root_dir

	def __len__(self):
		return len(self.poses)

	def __getitem__(self,idx):

		image_name = os.path.join(self.root_dir, self.poses[idx][0])
		print(image_name)
		image = cv2.imread(image_name)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype(np.float) / 255.0
		image = torch.from_numpy(image).type(torch.float)
		image = image.permute((2, 0, 1))

		pose = [math.sin(float(self.poses[idx][1]) * math.pi / 180.0), math.cos(float(self.poses[idx][1]) * math.pi / 180.0), \
					math.sin(float(self.poses[idx][2]) * math.pi / 180.0), math.cos(float(self.poses[idx][2]) * math.pi / 180.0)]
		pose = np.array(pose)
		pose = torch.Tensor(pose).type(torch.float)

		return {'image_path': image_name, 'image': image.to('cuda'), 'pose': pose.to('cuda')}


class ToTensor(object):
	def __call__(self,image):
		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2,0,1))
		return torch.from_numpy(image)



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.model_ft = models.resnet50(pretrained=True)

		for param in self.model_ft.parameters():
			param.requires_grad = True
		
		num_ftrs = self.model_ft.fc.in_features
		self.model_ft.fc = nn.Linear(num_ftrs, num_ftrs//2)
		self.fc1 = nn.Linear(num_ftrs//2, num_ftrs//4)
		self.fc2 = nn.Linear(num_ftrs//4, 4)


	def forward(self,x):
		x = func.tanh(self.model_ft(x))
		x = func.tanh(self.fc1(x))
		x = self.fc2(x)

		return x



def pose_loss(a,b,reduction='no_sum'):
	return(torch.norm(a-b, p=2))


def test(model,device,test_dataloader):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		f=open('eval_output.csv','a+')
		for i, sample in enumerate(test_dataloader):
			image_path, data, target = sample['image_path'], sample['image'], sample['pose']
			output = model(data)
			test_loss += pose_loss(output, target, reduction='sum').item() # sum up batch loss
			output = output.cpu().numpy()
			angles = np.arctan([output[0][0]/output[0][1], output[0][2]/output[0][3]])*(180/math.pi)
			image_name = image_path[0].split('/')
			f.write("%s, %1.3f, %1.3f\n" % (image_name[-1], angles[0], angles[1]))
		f.close()



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--train_csv', type=str, help='')
	parser.add_argument('--test_csv', type=str, help='')
	parser.add_argument('--images', type=str, help='')
	parser.add_argument('--epochs', type=int, help='')
	parser.add_argument('--save_dir', type=str, help='')
	parser.add_argument('--exp_name', type=str, help='')
	parser.add_argument('--eval_only', type=int, help='')
	parser.add_argument('--model_path',type=str, help='')
	args = parser.parse_args()

	tb = SummaryWriter()

	args = parser.parse_args()
	use_cuda = torch.cuda.is_available()

	device = torch.device("cuda" if use_cuda else "cpu")

	model = Net().to(device)
	

	if(args.eval_only == 0):

		train_dataset = ChairShapenetDataset(csv_file=args.train_csv, root_dir=args.images)
		train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)
		optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

		print(model)

		c = 0
		for epoch in range(0, args.epochs):

			model.train()
			with torch.enable_grad():
				for batch_idx, item in enumerate(train_dataloader):
					data, target = item['image'], item['pose']

					optimizer.zero_grad()
					output = model(data)
					loss = pose_loss(output, target)
					loss.backward()
					optimizer.step()

					tb.add_scalar('%s/train_loss' % args.exp_name, loss.item()/20, c)
					c += 1

			model.eval()
			test_loss = 0
			with torch.no_grad():
				for data, target in test_dataloader:
					data, target = item['image'], item['pose']

					output = model(data)
					test_loss += pose_loss(output, target, reduction='sum').item()

			test_loss /= len(test_dataloader.dataset)
			tb.add_scalar('%s/validation_loss' % args.exp_name, test_loss, epoch)

			if epoch % 2 == 0:
				torch.save(model.state_dict(),"%s/chair_pose_regressor_cnn_epoch_%s.pt" % (args.save_dir, epoch))


	if(args.eval_only == 1):
		test_dataset = ChairShapenetDataset(csv_file=args.test_csv, root_dir=args.images)
		test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
		model.load_state_dict(torch.load(args.model_path))
		test(model,device,test_dataloader)


#  test command
#  python pose_regressor.py --test_csv tests/test_list.csv --images tests/images/ --eval_only 1 --model_path chair_pose_regressor_cnn_epoch_368.pt