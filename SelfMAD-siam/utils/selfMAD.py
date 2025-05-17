import torch
from torchvision import utils
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import random
import cv2
import albumentations as alb
import argparse

import warnings
warnings.filterwarnings('ignore')

class selfMAD_Dataset(Dataset):
	def __init__(self,phase='train',image_size=256,datapath=None):
		# Check if this is a subclass that has already set up image_list, path_lm, and face_labels
		has_existing_data = hasattr(self, 'image_list') and hasattr(self, 'path_lm') and hasattr(self, 'face_labels')

		# If this is not a subclass or the subclass hasn't set up the data yet, initialize empty lists
		if not has_existing_data:
			self.path_lm = []
			self.face_labels = []
			self.image_list = []

			# Only load data if datapath is provided
			if datapath is not None:
				image_root = datapath
				landmark_root = image_root + "_landmarks"
				label_root = image_root + "_labels"
				if "FF++" in datapath:
					datapath = os.path.join(datapath, phase, "real")
					assert phase in ['train','val','test']
				elif any(method in datapath for method in ['FRLL', 'FRGC', 'FERET']):
					datapath = os.path.join(datapath, "raw")
				elif 'SMDD' in datapath:
					datapath = os.path.join(datapath, "os25k_bf_t")

				image_list, landmark_list, label_list = [], [], []
				for root, _, files in os.walk(datapath):
					for filename in files:
						if filename.endswith(('.png', '.jpg')):
							img_path = os.path.join(root, filename)
							landmark_path = img_path.replace(image_root, landmark_root, -1).replace('.png', '.npy').replace('.jpg', '.npy')
							label_path = img_path.replace(image_root, label_root, -1).replace('.png', '.npy').replace('.jpg', '.npy')
							if not os.path.isfile(landmark_path) or not os.path.isfile(label_path):
								continue
							image_list.append(img_path)
							landmark_list.append(landmark_path)
							label_list.append(label_path)

				self.path_lm = landmark_list
				self.face_labels = label_list
				self.image_list = image_list

		self.image_size=(image_size,image_size)
		self.phase=phase
		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()
		# print(len(self.image_list), len(self.path_lm), len(self.face_labels))

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		max_attempts = 5  # Limit the number of retries to avoid infinite loops
		attempts = 0

		while flag and attempts < max_attempts:
			try:
				filename=self.image_list[idx]

				# Normalize path (replace backslashes with forward slashes)
				filename = os.path.normpath(filename).replace('\\', '/')

				# Check if file exists
				if not os.path.exists(filename):
					# Try relative to current directory
					alt_path = os.path.join(".", filename)
					alt_path = os.path.normpath(alt_path).replace('\\', '/')
					if os.path.exists(alt_path):
						filename = alt_path
					else:
						print(f"Warning: Image file not found: {filename} or {alt_path}")
						raise FileNotFoundError(f"Image file not found: {filename}")

				# Load image
				try:
					img = np.array(Image.open(filename))
				except Exception as e:
					print(f"Error loading image {filename}: {str(e)}")
					raise

				# Load landmark
				landmark_path = self.path_lm[idx]
				landmark_path = os.path.normpath(landmark_path).replace('\\', '/')

				if not os.path.exists(landmark_path):
					# Try relative to current directory
					alt_path = os.path.join(".", landmark_path)
					alt_path = os.path.normpath(alt_path).replace('\\', '/')
					if os.path.exists(alt_path):
						landmark_path = alt_path
					else:
						print(f"Warning: Landmark file not found: {landmark_path} or {alt_path}")
						raise FileNotFoundError(f"Landmark file not found: {landmark_path}")

				try:
					landmark = np.load(landmark_path)
				except Exception as e:
					print(f"Error loading landmark {landmark_path}: {str(e)}")
					raise

				landmark = landmark[0] if len(landmark) == 1 else landmark # FF => (81, 2); SMDD (1, 81, 2)
				landmark = self.reorder_landmark(landmark)

				bbox = np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()]).reshape(2,2)

				# Load face label
				face_label_path = self.face_labels[idx]
				face_label_path = os.path.normpath(face_label_path).replace('\\', '/')

				if not os.path.exists(face_label_path):
					# Try relative to current directory
					alt_path = os.path.join(".", face_label_path)
					alt_path = os.path.normpath(alt_path).replace('\\', '/')
					if os.path.exists(alt_path):
						face_label_path = alt_path
					else:
						print(f"Warning: Face label file not found: {face_label_path} or {alt_path}")
						raise FileNotFoundError(f"Face label file not found: {face_label_path}")

				try:
					face_label = np.load(face_label_path)
				except Exception as e:
					print(f"Error loading face label {face_label_path}: {str(e)}")
					raise

				# random HFLIP
				if self.phase=='train':
					if np.random.rand()<0.5:
						img,_,landmark,bbox,face_label=self.hflip(img,None,landmark,bbox,face_label)

				img,landmark,bbox,face_label,_=self.crop_face(img,landmark,bbox,face_label,margin=True,crop_by_bbox=False,abs_coord=False,only_img=False)

				# BLENDING and MORPHING transforms
				if np.random.rand()<0.5:
					img_r,img_f, _ = self.self_blending(img.copy(),landmark.copy())
				else:
					img_r, img_f, _, _ = self.self_morphing(img.copy(), face_label.copy(), landmark.copy())

				if self.phase=='train':
					# Training augmentations
					transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
					img_f=transformed['image']
					img_r=transformed['image1']

					# Frequency transform
					if np.random.rand() < 0.1:
						freq_transform = self.create_frequency_noise_transform(weight=np.random.uniform(0.025, 0.1))
						img_f = freq_transform(image=img_f)['image']

				img_f,_,__,___, ____,y0_new,y1_new,x0_new,x1_new=self.crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
				img_r=img_r[y0_new:y1_new,x0_new:x1_new]

				img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255

				img_f=img_f.transpose((2,0,1))
				img_r=img_r.transpose((2,0,1))
				flag=False

			except Exception as e:
				print(f"Error processing item {idx}: {str(e)}")
				attempts += 1
				if attempts >= max_attempts:
					print(f"Maximum retry attempts reached for item {idx}. Creating fallback images.")
					# Create fallback images
					if isinstance(self.image_size, tuple):
						h, w = self.image_size
					else:
						h = w = self.image_size

					# Create blank images as fallback
					img_f = np.zeros((3, h, w), dtype=np.float32)
					img_r = np.zeros((3, h, w), dtype=np.float32)
					return img_f, img_r

				# Try a different random index
				idx = torch.randint(low=0, high=len(self), size=(1,)).item()
				print(f"Retrying with new index {idx}. Attempt {attempts}/{max_attempts}")

		return img_f, img_r

	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),

				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),

			], p=1.)


	def get_transforms(self):
		return alb.Compose([
			# Existing transforms
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),

			# New advanced transforms
			alb.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
			alb.GaussianBlur(blur_limit=(3, 7), p=0.1),
			alb.GridDistortion(p=0.1),
			alb.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),

			# Add color jitter for robustness to different lighting conditions
			alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
		],
		additional_targets={f'image1': 'image'},
		p=1.)

	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)

		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']

		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask


	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		mask=np.zeros_like(img[:,:,0])
		cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.randaffine(source,mask)

		img_blended,mask=self.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask

	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None,face_label=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)


			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]

		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None

		if face_label is not None:
			face_label=face_label[:,::-1]

		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new,face_label

	def crop_face(self,img,landmark=None,bbox=None,face_label=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
		H,W=len(img),len(img[0])
		if crop_by_bbox:
			x0,y0=bbox[0]
			x1,y1=bbox[1]
			w=x1-x0
			h=y1-y0
			w0_margin=w/4
			w1_margin=w/4
			h0_margin=h/4
			h1_margin=h/4
		else:
			x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
			x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
			w=x1-x0
			h=y1-y0
			w0_margin=w/8
			w1_margin=w/8
			h0_margin=h/2
			h1_margin=h/5
		if margin:
			w0_margin*=4
			w1_margin*=4
			h0_margin*=2
			h1_margin*=2
		elif phase=='train':
			w0_margin*=(np.random.rand()*0.6+0.2)
			w1_margin*=(np.random.rand()*0.6+0.2)
			h0_margin*=(np.random.rand()*0.6+0.2)
			h1_margin*=(np.random.rand()*0.6+0.2)
		else:
			w0_margin*=0.5
			w1_margin*=0.5
			h0_margin*=0.5
			h1_margin*=0.5

		y0_new=max(0,int(y0-h0_margin))
		y1_new=min(H,int(y1+h1_margin)+1)
		x0_new=max(0,int(x0-w0_margin))
		x1_new=min(W,int(x1+w1_margin)+1)

		img_cropped=img[y0_new:y1_new,x0_new:x1_new]

		if face_label is not None:
			face_label = np.stack([face_label, face_label, face_label], axis=-1)
			face_label_cropped=face_label[y0_new:y1_new,x0_new:x1_new]
			face_label_cropped = face_label_cropped[:,:,0]
		else:
			face_label_cropped = None
		if landmark is not None:
			landmark_cropped=np.zeros_like(landmark)
			for i,(p,q) in enumerate(landmark):
				landmark_cropped[i]=[p-x0_new,q-y0_new]
		else:
			landmark_cropped=None
		if bbox is not None:
			bbox_cropped=np.zeros_like(bbox)
			for i,(p,q) in enumerate(bbox):
				bbox_cropped[i]=[p-x0_new,q-y0_new]
		else:
			bbox_cropped=None

		if only_img:
			return img_cropped
		if abs_coord:
			return img_cropped,landmark_cropped,bbox_cropped,face_label_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
		else:
			return img_cropped,landmark_cropped,bbox_cropped,face_label_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)

	def create_frequency_noise_transform(self, weight):
		return alb.Compose([
			FrequencyPatterns(p=1.0, mode=4, required_pattern=5, weight=weight, max_multi=3)
		])

	def zoom_at_point(self, img, x, y, zoom_factor):
		height, width, _ = img.shape
		new_width = int(width / zoom_factor)
		new_height = int(height / zoom_factor)
		left = max(0, x - new_width // 2)
		top = max(0, y - new_height // 2)
		right = min(width, x + new_width // 2)
		bottom = min(height, y + new_height // 2)

		if left == 0:
			right = new_width
		if right == width:
			left = width - new_width
		if top == 0:
			bottom = new_height
		if bottom == height:
			top = height - new_height

		cropped_image_array = img[top:bottom, left:right]
		zoomed_image_array = cv2.resize(cropped_image_array, (width, height), interpolation=cv2.INTER_LANCZOS4)
		return zoomed_image_array

	def self_morph(self, img, face_labels, landmarks, default_config={'zoom_factor': 1.05, 'zoom_center': 'landmark_30', 'mask_type': [0, 18, 17], 'blend_factor': 0.5}, config={}):
		config = {**default_config, **config}

		# MASK
		mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
		for label in config['mask_type']:
			if label == "full":
				break
			mask = mask & (face_labels != label)

		# CENTER OF ZOOM
		if 'landmark' in config['zoom_center']:
			landmark_idx = int(config['zoom_center'].split('_')[-1])
			center = landmarks[landmark_idx, :]
			assert center is not None, 'no landmark center provided'
		elif config['zoom_center'] == 'mask_mean':
			center = np.mean(np.argwhere(mask == 1), axis=0)[::-1].astype(int)

		mask = np.stack([mask, mask, mask], axis=-1).astype(np.float32)
		source, target = img.copy(), img.copy()

		zoom_factor = config['zoom_factor']
		blend_factor = config['blend_factor']
		x, y = center

		zoomed_source = self.zoom_at_point(source, x, y,zoom_factor)
		zoomed_mask = self.zoom_at_point(mask, x, y, zoom_factor)

		img_f = target / 255
		zoomed_image_f = zoomed_source / 255

		morphed_img = np.where(zoomed_mask == 0, img_f, (1 - blend_factor) * img_f + blend_factor * zoomed_image_f)
		morphed_img = (morphed_img * 255).astype(np.uint8)
		bonafide_img = (img_f * 255).astype(np.uint8)

		return bonafide_img, morphed_img, center

	def self_morphing(self, img, face_labels, landmarks):
		config = {
			'zoom_factor': np.random.uniform(1.0, 1.1),
			'zoom_center': random.choice(['mask_mean', 'landmark_30']),
			'mask_type': random.choice([[0, 18, 17], [0, 13, 17, 18], [0, 13, 17, 18, 8, 9], [0, 17, 18, 8, 9],
										[0, 17], [0, 13, 17], [0, 13, 17, 8, 9], [0, 17, 8, 9],
										[18, 17], [13, 17, 18], [13, 17, 18, 8, 9], [17, 18, 8, 9],
										[17], [13, 17], [13, 17, 8, 9], [17, 8, 9],
										[0, 18], [0, 13, 18], [0, 13, 18, 8, 9], [0, 18, 8, 9],
										[0], [0, 13], [0, 13, 8, 9], [0, 8, 9],
										[18], [13, 18], [13, 18, 8, 9], [18, 8, 9],
										[13], [13, 8, 9], [8, 9],
										"full"
										]),
			'blend_factor': random.choice([0.5, 0.5, 0.5, 0.375, 0.25, 0.125])
		}
		img_bonafide, morphed_img, center = self.self_morph(img, face_labels, landmarks, config)
		return img_bonafide, morphed_img, center, config

	def dynamic_blend(self, source,target,mask):
		mask_blured = self.get_blend_mask(mask)
		blend_list=[0.25,0.5,0.75,1,1,1]
		blend_ratio = blend_list[np.random.randint(len(blend_list))]
		mask_blured*=blend_ratio
		img_blended=(mask_blured * source + (1 - mask_blured) * target)
		return img_blended,mask_blured

	def get_blend_mask(self, mask):
		H,W=mask.shape
		size_h=np.random.randint(192,257)
		size_w=np.random.randint(192,257)
		mask=cv2.resize(mask,(size_w,size_h))
		kernel_1=random.randrange(5,26,2)
		kernel_1=(kernel_1,kernel_1)
		kernel_2=random.randrange(5,26,2)
		kernel_2=(kernel_2,kernel_2)

		mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
		mask_blured = mask_blured/(mask_blured.max())
		mask_blured[mask_blured<1]=0

		mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
		mask_blured = mask_blured/(mask_blured.max())
		mask_blured = cv2.resize(mask_blured,(W,H))
		return mask_blured.reshape((mask_blured.shape+(1,)))

	def collate_fn(self,batch):
		"""
		Collate function for DataLoader.

		Note: Images from __getitem__ are already normalized to [0,1] and in format (C,H,W).
		This function converts them to tensors and ensures they're in the correct format.
		"""
		img_f,img_r=zip(*batch)
		data={}

		try:
			# Convert images to tensors and ensure they're in the correct format (B,C,H,W)
			img_r_array = np.array(img_r)
			img_f_array = np.array(img_f)

			# Check if the arrays are already in the correct format (B,C,H,W)
			if len(img_r_array.shape) == 4:
				if img_r_array.shape[1] == 3:  # Already in (B,C,H,W) format
					img_r_tensor = torch.tensor(img_r_array).float()
					img_f_tensor = torch.tensor(img_f_array).float()
				elif img_r_array.shape[3] == 3:  # In (B,H,W,C) format, need to transpose
					img_r_tensor = torch.tensor(img_r_array).float().permute(0, 3, 1, 2)
					img_f_tensor = torch.tensor(img_f_array).float().permute(0, 3, 1, 2)
				else:
					# Try to handle other formats
					if img_r_array.shape[2] == 3:  # Possibly (B,H,C,W) format
						img_r_tensor = torch.tensor(img_r_array).float().permute(0, 2, 1, 3)
						img_f_tensor = torch.tensor(img_f_array).float().permute(0, 2, 1, 3)
					else:
						raise ValueError(f"Unexpected array shape: {img_r_array.shape}")
			else:
				# If not a 4D array, try to convert it
				img_r_tensor = torch.tensor(img_r_array).float()
				img_f_tensor = torch.tensor(img_f_array).float()

				# Ensure we have a 4D tensor with channels in the right position
				if img_r_tensor.dim() < 4:
					if img_r_tensor.dim() == 3:
						if img_r_tensor.shape[0] == 3:  # Single image in (C,H,W) format
							# Add batch dimension
							img_r_tensor = img_r_tensor.unsqueeze(0)
							img_f_tensor = img_f_tensor.unsqueeze(0)
						elif img_r_tensor.shape[2] == 3:  # Single image in (H,W,C) format
							# Add batch dimension and reorder to (B,C,H,W)
							img_r_tensor = img_r_tensor.permute(2, 0, 1).unsqueeze(0)
							img_f_tensor = img_f_tensor.permute(2, 0, 1).unsqueeze(0)

			# Ensure pixel values are in the range [0, 1]
			if img_r_tensor.max() > 1.0:
				img_r_tensor = img_r_tensor / 255.0
			if img_f_tensor.max() > 1.0:
				img_f_tensor = img_f_tensor / 255.0

			# Final check to ensure tensors have the right shape (B,C,H,W)
			if img_r_tensor.dim() != 4 or img_r_tensor.shape[1] != 3:
				raise ValueError(f"Failed to convert tensors to the correct format. Current shape: {img_r_tensor.shape}")

			# Concatenate along batch dimension
			data['img'] = torch.cat([img_r_tensor, img_f_tensor], 0)
			data['label'] = torch.tensor([0]*len(img_r)+[1]*len(img_f))

			return data
		except Exception as e:
			# Fallback to simpler implementation if the complex one fails
			try:
				# Simple implementation that works for most cases
				img_r_tensor = torch.tensor(np.array(img_r)).float()
				img_f_tensor = torch.tensor(np.array(img_f)).float()

				# Ensure pixel values are in the range [0, 1]
				if img_r_tensor.max() > 1.0:
					img_r_tensor = img_r_tensor / 255.0
				if img_f_tensor.max() > 1.0:
					img_f_tensor = img_f_tensor / 255.0

				data['img'] = torch.cat([img_r_tensor, img_f_tensor], 0)
				data['label'] = torch.tensor([0]*len(img_r)+[1]*len(img_f))

				return data
			except Exception as inner_e:
				# If all else fails, raise the original error
				raise ValueError(f"Failed to process batch: {e}. Inner error: {inner_e}")

	def worker_init_fn(self,worker_id):
		np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__=='__main__':
	from albu import FrequencyPatterns,RandomDownScale

	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	parser=argparse.ArgumentParser()
	parser.add_argument('--datapath',type=str, required=True)
	args=parser.parse_args()
	datapath=args.datapath

	img_size=256
	batch_size=20
	image_dataset=selfMAD_Dataset(phase='train',image_size=img_size, datapath=datapath)

	dataloader = torch.utils.data.DataLoader(image_dataset,
					batch_size=batch_size,
					shuffle=True,
					collate_fn=image_dataset.collate_fn,
					num_workers=0,
					worker_init_fn=image_dataset.worker_init_fn
					)

	data_iter=iter(dataloader)
	data=next(data_iter)
	img=data['img']
	img=img.view((-1,3,img_size,img_size))
	utils.save_image(img, 'SelfMAD_transform_example.png', nrow=batch_size, normalize=False)
	print("image saved")

else:
	try:
		# Try relative import first
		from utils.albu import FrequencyPatterns, RandomDownScale
	except ImportError:
		try:
			# Try absolute import from SelfMAD-siam
			# Try with different module name
			from SelfMAD_siam.utils.albu import FrequencyPatterns, RandomDownScale
		except ImportError:
			try:
				# Try with current directory
				import sys
				import os
				sys.path.append(os.path.dirname(os.path.abspath(__file__)))
				from albu import FrequencyPatterns, RandomDownScale
			except ImportError as e:
				print(f"Error importing FrequencyPatterns and RandomDownScale: {e}")
				print("Using fallback implementations")

				# Fallback implementations if imports fail
				class FrequencyPatterns:
					def __init__(self, p=0.5, required_pattern=0, max_multi=1, mode=None, weight=None):
						self.p = p

					def __call__(self, **kwargs):
						return kwargs

				class RandomDownScale:
					def __init__(self, p=0.5):
						self.p = p

					def __call__(self, **kwargs):
						return kwargs
