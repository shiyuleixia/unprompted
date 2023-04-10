from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import draw_segmentation_masks
import torch
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from modules.processing import process_images,Processed, StableDiffusionProcessingImg2Img

class Shortcode():
	def __init__(self,Unprompted):
		self.image_mask_combine = None
		self.Unprompted = Unprompted
		self.image_mask = None
		self.image_masks = None
		self.show = False
		self.per_instance = False
		self.description = "Creates an image mask from instances of types specified by the content for use with inpainting."

	def run_block(self, pargs, kwargs, context, content):
		from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
		from kornia.morphology import dilation, erosion
		from kornia.filters import box_blur

		if "init_images" not in self.Unprompted.shortcode_user_vars:
			return

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


		self.show = True if "show" in pargs else False
		self.per_instance = True if "per_instance" in pargs else False

		brush_mask_mode = self.Unprompted.parse_advanced(kwargs["mode"],context) if "mode" in kwargs else "add"
		select_mode = self.Unprompted.parse_advanced(kwargs["select_mode"],context) if "select_mode" in kwargs else "overlap"

		smoothing_kernel = None
		smoothing = int(self.Unprompted.parse_advanced(kwargs["smoothing"],context)) if "smoothing" in kwargs else 20

		if smoothing > 0:
			smoothing_kernel = torch.ones(1, smoothing, smoothing, device=device)/(smoothing**2)

		# Pad the mask by applying a dilation or erosion
		mask_padding = int(self.Unprompted.parse_advanced(kwargs["padding"],context) if "padding" in kwargs else 0)
		padding_dilation_kernel = None
		if (mask_padding != 0):
			padding_dilation_kernel = torch.ones(abs(mask_padding), abs(mask_padding), device=device)

		prompts = content.split(self.Unprompted.Config.syntax.delimiter)
		prompt_parts = len(prompts)

		mask_precision = min(1.0,float(self.Unprompted.parse_advanced(kwargs["mask_precision"],context) if "mask_precision" in kwargs else 0.5))
		instance_precision = min(1.0,float(self.Unprompted.parse_advanced(kwargs["instance_precision"],context) if "instance_precision" in kwargs else 0.85))
		num_instances = int(self.Unprompted.parse_advanced(kwargs["select"],context) if "select" in kwargs else 0)

		init_image = self.Unprompted.shortcode_user_vars["init_images"][0]

		masks = self.Unprompted.shortcode_user_vars.setdefault("image_mask", None)
		if masks is not None:
			masks = pil_to_tensor(self.Unprompted.shortcode_user_vars["image_mask"].convert('L').resize((512, 512))) > 0
		else:
			masks = torch.zeros(512, 512, dtype=torch.bool)
		
		weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
		transforms = weights.transforms()
		model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False).eval().to(device=device)

		image = init_image
		image = init_image.resize((512, 512))
		image = transforms(image)

		pred = model(image[None].to(device=device))[0]

		target_labels = [weights.meta["categories"].index(i) for i in prompts]
		wanted_masks = torch.tensor([label in target_labels for label in pred["labels"]], device=device)
		likely_masks = (pred["scores"] > instance_precision)
		instance_masks:torch.Tensor = pred["masks"][likely_masks & wanted_masks]

		instance_masks = instance_masks.float()

		if mask_padding != 0:
			if mask_padding > 0:
				instance_masks = dilation(instance_masks, kernel=padding_dilation_kernel)
			else:
				instance_masks = erosion(instance_masks, kernel=padding_dilation_kernel)
		
		if smoothing != 0:
			instance_masks = box_blur(instance_masks, (smoothing, smoothing))

		instance_masks = instance_masks > mask_precision
		instance_masks = instance_masks > 0
		instance_masks = instance_masks.cpu()

		if num_instances > 0:
			if "overlap" in select_mode:
				# select the instance with the highest overlay on the mask
				mask_in_instance = masks[None].broadcast_to(instance_masks.shape).clone()
				# count only parts of mask that are in instance mask
				mask_in_instance[~instance_masks] = 0
				overlap = mask_in_instance.count_nonzero(dim=[1,2,3])
				
				if select_mode == "relative overlap":
					overlap = overlap / instance_masks.count_nonzero(dim=[1,2,3])
					
				val, idx = torch.topk(overlap, k=num_instances)	
				instance_masks = instance_masks[idx]

			elif select_mode == "greatest area":
				# select the instance with the greatest mask
				val, idx = torch.topk(instance_masks.count_nonzero(dim=[1,2,3]), k=num_instances)
				instance_masks = instance_masks[idx]
			elif select_mode == "random":
				idx = torch.randperm(len(instance_masks))[:num_instances]
				instance_masks = instance_masks[idx]

		if num_instances > 0:
			instance_masks = instance_masks.sum(dim=0)
		else:
			instance_masks = instance_masks.squeeze(dim=1)

		masks = masks.broadcast_to(instance_masks.shape).clone()
		if brush_mask_mode == "refine":
			refine_mask = instance_masks > 0
			masks[~refine_mask] = 0
		elif brush_mask_mode == "add":
			masks = masks + instance_masks
			masks = masks > 0
		elif brush_mask_mode == "subtract":
			masks = ((instance_masks > 0) & ~masks)
		elif brush_mask_mode == "discard":
			masks = instance_masks > 0

		# remove empty masks
		masks = masks[masks.count_nonzero(dim=[1,2]) != 0]

		if self.per_instance:
			# support multiple will draw the other instances
			self.image_mask = to_pil_image(masks[0].float()).resize((init_image.width, init_image.height))
			self.Unprompted.log("instance2mask init_image size:{}:{}".format(init_image.width,init_image.height))
			# save instance masks for support_multiple to pick it up
			self.Unprompted.shortcode_user_vars["image_masks"] = [to_pil_image(m.float()).resize((init_image.width, init_image.height)) for m in masks]
			self.image_masks = self.Unprompted.shortcode_user_vars["image_masks"]
			combined_mask = masks.sum(dim=0, keepdim=True) > 0
			self.image_mask_combine = to_pil_image(combined_mask.float()).resize((init_image.width, init_image.height))
			self.Unprompted.shortcode_user_vars["image_masks_combine"] = self.image_mask_combine

		else:
			combined_mask = masks.sum(dim=0, keepdim=True) > 0
			self.image_mask = to_pil_image(combined_mask.float()).resize((init_image.width, init_image.height))
			# store instance masks for later segmentation drawing
			self.image_masks = [to_pil_image(m.float()).resize((init_image.width, init_image.height)) for m in masks]
			self.Unprompted.shortcode_user_vars["image_masks"] = [self.image_mask]	

		self.Unprompted.shortcode_user_vars["mode"] = 1
		self.Unprompted.shortcode_user_vars["mask_mode"] = 1
		self.Unprompted.shortcode_user_vars["image_mask"] = self.image_mask
		self.Unprompted.shortcode_user_vars["mask_for_overlay"] = self.image_mask
		self.Unprompted.shortcode_user_vars["latent_mask"] = None # fixes inpainting full resolution

		if "save" in kwargs: self.image_mask.save(f"{self.Unprompted.parse_advanced(kwargs['save'],context)}.png")

		return ""
	
	def after(self, p:StableDiffusionProcessingImg2Img, processed:Processed):
		if self.image_masks and self.show:
			image = pil_to_tensor(p.init_images[-1])
			
			masks = torch.stack([pil_to_tensor(m) for m in self.image_masks]).squeeze(dim=1)
			image = draw_segmentation_masks(image, masks > 0, alpha=0.75)
			processed.images += self.image_masks + [to_pil_image(image)]
			self.image_masks = None

		return processed
	
	def ui(self,gr):
		gr.Radio(label="Mask blend mode 🡢 mode",choices=["add","subtract","discard", "refine"], value="add", interactive=True)
		gr.Checkbox(label="Show mask in output 🡢 show")
		gr.Checkbox(label="Run inpaint per instance found 🡢 per_instance")
		gr.Number(label="Precision of selected area 🡢 mask_precision",value=0.5,interactive=True)
		gr.Number(label="Padding radius in pixels 🡢 padding",value=0,interactive=True)
		gr.Number(label="Smoothing radius in pixels 🡢 smoothing",value=20,interactive=True)
		gr.Number(label="Precision of instance selection 🡢 instance_precision",value=0.85,interactive=True)
		gr.Number(label="Number of instance to select 🡢 select",value=0,interactive=True)
		gr.Radio(
			label="Instance selection mode 🡢 select_mode",
			choices=["overlap", "relative overlap", "random", "greatest area"], 
			value="overlap",
			interactive=True
		)