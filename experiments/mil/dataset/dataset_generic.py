from __future__ import print_function, division
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

from torch.utils.data import Dataset
import h5py

# from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path,
		shuffle = False, 
		seed = 42, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		selected_unit_indices: Optional[np.ndarray] = None,
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col
		self.selected_unit_indices = selected_unit_indices

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		print(slide_data)
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()
		
		self.data_cache = {}

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	# def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
	# 	settings = {
	# 				'n_splits' : k, 
	# 				'val_num' : val_num, 
	# 				'test_num': test_num,
	# 				'label_frac': label_frac,
	# 				'seed': self.seed,
	# 				'custom_test_ids': custom_test_ids
	# 				}

	# 	if self.patient_strat:
	# 		settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
	# 	else:
	# 		settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

	# 	self.split_gen = generate_split(**settings)

	# def set_splits(self,start_from=None):
	# 	if start_from:
	# 		ids = nth(self.split_gen, start_from)

	# 	else:
	# 		ids = next(self.split_gen)

	# 	if self.patient_strat:
	# 		slide_ids = [[] for i in range(len(ids))] 

	# 		for split in range(len(ids)): 
	# 			for idx in ids[split]:
	# 				case_id = self.patient_data['case_id'][idx]
	# 				slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
	# 				slide_ids[split].extend(slide_indices)

	# 		self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

	# 	else:
	# 		self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, backbone, patch_size, all_splits, split_key='train', selected_unit_indices: Optional[np.ndarray] = None):
		split_df_ids = all_splits[split_key]
		split_df_ids = split_df_ids.dropna().reset_index(drop=True)

		if len(split_df_ids) > 0:
			mask = self.slide_data['slide_id'].isin(split_df_ids.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			# Pass selected_unit_indices to Generic_Split
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, selected_unit_indices=selected_unit_indices)
			split.set_backbone(backbone)
			split.set_patch_size(patch_size)
		else:
			split = None
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split

	def return_splits(self, backbone: str, patch_size: str = '', from_id: bool = True, csv_path: Optional[str] = None, selected_unit_indices: Optional[np.ndarray] = None): # Added selected_unit_indices
		if from_id:
			if self.train_ids is not None and len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				# Pass selected_unit_indices to Generic_Split
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes, selected_unit_indices=selected_unit_indices)
				train_split.set_backbone(backbone)
				train_split.set_patch_size(patch_size)
			else:
				train_split = None

			if self.val_ids is not None and len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes, selected_unit_indices=selected_unit_indices)
				val_split.set_backbone(backbone)
				val_split.set_patch_size(patch_size)
			else:
				val_split = None
			
			if self.test_ids is not None and len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes, selected_unit_indices=selected_unit_indices)
				test_split.set_backbone(backbone)
				test_split.set_patch_size(patch_size)
			else:
				test_split = None
		else:
			assert csv_path
			all_splits_df = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
			# Pass selected_unit_indices to get_split_from_df
			train_split = self.get_split_from_df(backbone, patch_size, all_splits_df, 'train', selected_unit_indices=selected_unit_indices)
			val_split = self.get_split_from_df(backbone, patch_size, all_splits_df, 'val', selected_unit_indices=selected_unit_indices)
			test_split = self.get_split_from_df(backbone, patch_size, all_splits_df, 'test', selected_unit_indices=selected_unit_indices)
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	# def save_split(self, filename):
	# 	train_split = self.get_list(self.train_ids)
	# 	val_split = self.get_list(self.val_ids)
	# 	test_split = self.get_list(self.test_ids)
	# 	df_tr = pd.DataFrame({'train': train_split})
	# 	df_v = pd.DataFrame({'val': val_split})
	# 	df_t = pd.DataFrame({'test': test_split})
	# 	df = pd.concat([df_tr, df_v, df_t], axis=1) 
	# 	df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False
		self.backbone=None
		self.patch_size=''
		self.data_cache = {}

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		elif self.data_dir is None:
			data_dir = self.slide_data['dir'][idx]
		else:
			data_dir = self.data_dir
		

		# Determine data_dir (refined logic for current_data_dir)
		current_data_dir = None
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			current_data_dir = self.data_dir[source]
		elif self.data_dir is None:
			if 'dir' in self.slide_data.columns:
				current_data_dir = self.slide_data['dir'][idx] # Corrected indentation
			else: # Fallback for Generic_Split
				current_data_dir = self.data_dir # Corrected indentation; self.data_dir was passed to Generic_Split
		else:
			current_data_dir = self.data_dir
		
		if current_data_dir is None:
			raise ValueError(f"Could not determine data directory for slide_id {slide_id}")

		if not self.use_h5:
			if self.backbone is None:
				raise ValueError("Backbone not set. Call set_backbone() before loading data.")

			pt_files_dir = Path(current_data_dir) / self.backbone / 'pt_files'
			if self.patch_size and self.patch_size != "": # Consider patch_size for path
				pt_files_dir = pt_files_dir / str(self.patch_size)
			
			full_path = str(pt_files_dir / f'{slide_id}.pt')

			if full_path in self.data_cache:
				features = self.data_cache[full_path]
			else:
				if not Path(full_path).exists():
					raise FileNotFoundError(f"Feature file not found: {full_path}")
				features = torch.load(full_path)
				
				# Slice features if selected_unit_indices is available
				if self.selected_unit_indices is not None:
					if not isinstance(features, torch.Tensor):
						raise TypeError(f"Expected features to be a torch.Tensor, but got {type(features)}")
					if features.ndim != 2: # Expect (N_patches, N_features)
						raise ValueError(f"Expected features to be 2D (N_patches, N_features), but got shape {features.shape}") # Corrected indentation
					
					num_original_features = features.shape[1]
					if np.any(self.selected_unit_indices >= num_original_features):
						max_selected_idx = np.max(self.selected_unit_indices)
						raise ValueError(
							f"Max selected unit index ({max_selected_idx}) is out of bounds "
							f"for features with {num_original_features} dimensions (from file {full_path})."
						)
					features = features[:, self.selected_unit_indices]

				# Cache if cache_flag is set (typically by Generic_Split.pre_loading)
				if hasattr(self, 'cache_flag') and self.cache_flag:
					self.data_cache[full_path] = features
			return features, label
			
			# else:
			# 	return slide_id, label

		else: # H5 loading
			full_path = os.path.join(current_data_dir,'h5_files','{}.h5'.format(slide_id)) # Use current_data_dir
			if not Path(full_path).exists():
				raise FileNotFoundError(f"H5 feature file not found: {full_path}")
			with h5py.File(full_path,'r') as hdf5_file:
				features_np = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			# Slice H5 features if selected_unit_indices is available
			if self.selected_unit_indices is not None:
				if features_np.ndim != 2:
					raise ValueError(f"Expected H5 features to be 2D (N_patches, N_features), but got shape {features_np.shape}")
				num_original_features = features_np.shape[1]
				if np.any(self.selected_unit_indices >= num_original_features):
					max_selected_idx = np.max(self.selected_unit_indices)
					raise ValueError(
						f"Max selected unit index ({max_selected_idx}) is out of bounds "
						f"for H5 features with {num_original_features} dimensions (from file {full_path})."
					)
				features_np = features_np[:, self.selected_unit_indices]

			features = torch.from_numpy(features_np)
			return features, label, coords

	def set_backbone(self, backbone):
		self.backbone = backbone
	
	def set_patch_size(self, size):
		self.patch_size = size


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2, selected_unit_indices: Optional[np.ndarray] = None):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.selected_unit_indices = selected_unit_indices
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		self.data_cache = {}
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		
	def set_backbone(self, backbone):
		print('Setting Backbone:', backbone)
		self.backbone = backbone

	def set_patch_size(self, size):
		print('Setting Patchsize:', size)
		self.patch_size = size

	def pre_loading(self, thread=8):
		# set flag
		self.cache_flag = True

		ids = list(range(len(self)))
		from multiprocessing.pool import ThreadPool
		exe = ThreadPool(thread)
		exe.map(self.__getitem__, ids)