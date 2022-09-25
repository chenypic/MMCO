import math
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd

from CLAM.wsi_core.batch_process_utils import initialize_df
from tiatoolbox.wsicore.wsireader import WSIReader
import cv2


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,default='/data/ichenwei/TCGA-HE-OV-end/source/',
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=True, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,default='/data/ichenwei/TCGA-HE-OV-end/MMCO/',
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')


def segment(WSI_object,seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False, 
				filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[],wsi_info=None):

	def _filter_contours(contours, hierarchy, filter_params):
		"""
			Filter contours by: area.
		"""
		filtered = []

		# find indices of foreground contours (parent == -1)
		hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
		all_holes = []
					
		# loop through foreground contour indices
		for cont_idx in hierarchy_1:
			# actual contour
			cont = contours[cont_idx]
			# indices of holes contained in this contour (children of parent contour)
			holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
			# take contour area (includes holes)
			a = cv2.contourArea(cont)
			# calculate the contour area of each hole
			hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
			# actual area of foreground contour region
			a = a - np.array(hole_areas).sum()
			if a == 0: continue
			if tuple((filter_params['a_t'],)) < tuple((a,)): 
				filtered.append(cont_idx)
				all_holes.append(holes)


		foreground_contours = [contours[cont_idx] for cont_idx in filtered]
		
		hole_contours = []

		for hole_ids in all_holes:
			unfiltered_holes = [contours[idx] for idx in hole_ids ]
			unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
			# take max_n_holes largest holes by area
			unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
			filtered_holes = []
						
			# filter these holes
			for hole in unfilered_holes:
				if cv2.contourArea(hole) > filter_params['a_h']:
					filtered_holes.append(hole)

			hole_contours.append(filtered_holes)

		return foreground_contours, hole_contours

	
	number_contours = False
	
	img = np.array(WSI_object.read_rect((0,0), wsi_info['level_dimensions'][seg_level], resolution=seg_level, units='level'))

	print(seg_level)
	print(img.shape)
	print(wsi_info['level_dimensions'])

	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Convert to HSV space
	img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring

	# Thresholding
	if use_otsu:
		_, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
	else:
		_, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

	# Morphological closing
	if close > 0:
		kernel = np.ones((close, close), np.uint8)
		img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

		
	scale = int(wsi_info['level_downsamples'][seg_level])
	scaled_ref_patch_area = int(ref_patch_size**2 / (scale * scale))
	filter_params = filter_params.copy()
	filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
	filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

	# Find and filter contours
	contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
	hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
	if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts

	#self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
	#self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

	contours_tissue = [np.array(cont * scale, dtype='int32') for cont in contours]
	holes_tissue = [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

	#exclude_ids = [0,7,9]
	if len(keep_ids) > 0:
		contour_ids = set(keep_ids) - set(exclude_ids)
	else:
		contour_ids = set(np.arange(len(contours_tissue))) - set(exclude_ids)

	contours_tissue = [contours_tissue[i] for i in contour_ids]
	holes_tissue = [holes_tissue[i] for i in contour_ids]

	top_left = (0,0)
	region_size = wsi_info['level_dimensions'][current_vis_params['vis_level']]

	print('region_size:',region_size)

	img = WSI_object.read_region(top_left, current_vis_params['vis_level'], region_size)

	offset = tuple(-(np.array(top_left) * scale).astype(int))
	line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
	if contours_tissue is not None:
		if not number_contours:
			cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale), 
								-1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)







if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	print(directories['source'])

	slides = sorted(os.listdir(directories['source']))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(directories['source'], slide))]

	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in range(total):
		df.to_csv(os.path.join(directories['save_dir'], 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if args.no_auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		full_path = os.path.join(directories['source'], slide)
		WSI_object = WSIReader.open(input_img=full_path)
		wsi_info = WSI_object.info.as_dict()



		current_vis_params = {}
		current_filter_params = {}
		current_seg_params = {}
		current_patch_params = {}



		for key in vis_params.keys():
			if legacy_support and key == 'vis_level':
				df.loc[idx, key] = -1
			current_vis_params.update({key: df.loc[idx, key]})

		for key in filter_params.keys():
			if legacy_support and key == 'a_t':
				old_area = df.loc[idx, 'a']
				seg_level = df.loc[idx, 'seg_level']
				scale = wsi_info['level_downsamples'][seg_level]
				adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
				current_filter_params.update({key: adjusted_area})
				df.loc[idx, key] = adjusted_area
			current_filter_params.update({key: df.loc[idx, key]})

		for key in seg_params.keys():
			if legacy_support and key == 'seg_level':
				df.loc[idx, key] = -1
			current_seg_params.update({key: df.loc[idx, key]})

		for key in patch_params.keys():
			current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(wsi_info['level_dimensions']) == 1:
				current_vis_params['vis_level'] = 0
			
			else:
				best_level = WSI_object.openslide_wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(wsi_info['level_dimensions']) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				best_level = WSI_object.openslide_wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = wsi_info['level_dimensions'][current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		print(current_filter_params)

		seg_time_elapsed = -1
		if args.seg:
			#WSI_object, seg_time_elapsed = segment(WSI_object, **current_seg_params, filter_params=current_filter_params) 

			segment(WSI_object, **current_seg_params, filter_params=current_filter_params,wsi_info=wsi_info) 











