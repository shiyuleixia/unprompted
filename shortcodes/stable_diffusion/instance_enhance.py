import PIL.ImageOps


class Shortcode():
    def __init__(self, Unprompted):
        self.Unprompted = Unprompted
        self.description = "Upscales a selected portion of the image. ENHANCE!"
        self.is_fixing = False
        self.wizard_prepend = Unprompted.Config.syntax.tag_start + "after" + Unprompted.Config.syntax.tag_end + Unprompted.Config.syntax.tag_start_alt + "zoom_enhance"
        self.wizard_append = Unprompted.Config.syntax.tag_end_alt + Unprompted.Config.syntax.tag_start + Unprompted.Config.syntax.tag_close + "after" + Unprompted.Config.syntax.tag_end
        self.resample_methods = {}
        self.resample_methods["Nearest Neighbor"] = 0
        self.resample_methods["Box"] = 4
        self.resample_methods["Bilinear"] = 2
        self.resample_methods["Hamming"] = 5
        self.resample_methods["Bicubic"] = 3
        self.resample_methods["Lanczos"] = 1

    def run_atomic(self, pargs, kwargs, context):
        import cv2
        import numpy
        from PIL import Image, ImageFilter
        import math
        import traceback

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        blur_radius_orig = float(
            self.Unprompted.parse_advanced(kwargs["blur_size"], context)) if "blur_size" in kwargs else 0.03
        upscale_width = int(float(
            self.Unprompted.parse_advanced(kwargs["upscale_width"], context))) if "upscale_width" in kwargs else 512
        upscale_height = int(float(
            self.Unprompted.parse_advanced(kwargs["upscale_height"], context))) if "upscale_height" in kwargs else 512
        save = True if "save" in pargs else False
        random = True if "random" in pargs else False
        sort_by_centroid = True if "sort_by_centroid" in pargs else False
        ema_face = True if "ema_face" in pargs else False
        ema_face_dialate = True if "ema_face_dialate" in pargs else False
        instance_dialate = True if "instance_dialate" in pargs else False
        img2img = True if "img2img" in pargs else False
        use_workaround = True if "use_workaround" in pargs else False
        mask_sort_method = self.Unprompted.parse_alt_tags(kwargs["mask_sort_method"],
                                                          context) if "mask_sort_method" in kwargs else "left-to-right"
        downscale_method = self.Unprompted.parse_alt_tags(kwargs["downscale_method"],
                                                          context) if "downscale_method" in kwargs else "Lanczos"
        downscale_method = self.resample_methods[downscale_method]
        upscale_method = self.Unprompted.parse_alt_tags(kwargs["upscale_method"],
                                                        context) if "downscale_method" in kwargs else "Nearest Neighbor"
        upscale_method = self.resample_methods[upscale_method]

        all_replacements = (self.Unprompted.parse_alt_tags(kwargs["replacement"],
                                                           context) if "replacement" in kwargs else "face").split(
            self.Unprompted.Config.syntax.delimiter)
        #all_negative_replacements = (self.Unprompted.parse_alt_tags(kwargs["negative_replacement"],
        #                                                            context) if "negative_replacement" in kwargs else "").split(
        #    self.Unprompted.Config.syntax.delimiter)
        
        image_cfg_scale = float(self.Unprompted.parse_advanced(kwargs["image_cfg_scale"],context)) \
                if "image_cfg_scale" in kwargs else 1.5

        self.Unprompted.shortcode_user_vars["width"] = upscale_width
        self.Unprompted.shortcode_user_vars["height"] = upscale_height
        # Ensure standard img2img mode
        self.Unprompted.shortcode_user_vars["mode"] = 0
        self.Unprompted.shortcode_user_vars["image_mask"] = None
        self.Unprompted.shortcode_user_vars["mask"] = None
        self.Unprompted.shortcode_user_vars["init_img_with_mask"] = None
        self.Unprompted.shortcode_user_vars["init_mask"] = None
        self.Unprompted.shortcode_user_vars["mask_mode"] = 0
        self.Unprompted.shortcode_user_vars["init_mask_inpaint"] = None
        self.Unprompted.shortcode_user_vars["inpaint_full_res"] = 1
        self.Unprompted.shortcode_user_vars["inpaint_full_res_padding"] = 32
        self.Unprompted.shortcode_user_vars["mask_blur"] = 4
        inpainting_fill = int(self.Unprompted.parse_advanced(kwargs["inpainting_fill"],context)) if "inpainting_fill" in kwargs else 1
        self.Unprompted.shortcode_user_vars["resize_mode"] = 0
        self.Unprompted.shortcode_user_vars["img2img_init_image"] = None
        self.Unprompted.shortcode_user_vars["image_cfg_scale"] = image_cfg_scale
        #self.Unprompted.shortcode_user_vars["seed"] = -1
        # Prevent batch from breaking
        batch_size_orig = self.Unprompted.shortcode_user_vars["batch_size"]
        n_iter_orig = self.Unprompted.shortcode_user_vars["n_iter"]
        self.Unprompted.shortcode_user_vars["batch_size"] = 1
        self.Unprompted.shortcode_user_vars["n_iter"] = 1

        if "denoising_strength" in kwargs: self.Unprompted.shortcode_user_vars["denoising_strength"] = float(
            self.Unprompted.parse_advanced(kwargs["denoising_strength"], context))
        if "cfg_scale" in kwargs: self.Unprompted.shortcode_user_vars["cfg_scale"] = float(
            self.Unprompted.parse_advanced(kwargs["cfg_scale"], context))

        # vars for dynamic settings
        denoising_max = float(
            self.Unprompted.parse_advanced(kwargs["denoising_max"], context)) if "denoising_max" in kwargs else 0.65
        cfg_min = float(
            self.Unprompted.parse_advanced(kwargs["cfg_scale_min"], context)) if "cfg_scale_min" in kwargs else 7.0
        target_size_max = float(
            self.Unprompted.parse_advanced(kwargs["mask_size_max"], context)) if "mask_size_max" in kwargs else 0.3
        cfg_max = self.Unprompted.shortcode_user_vars["cfg_scale"] - cfg_min

        padding_original = int(float(
            self.Unprompted.parse_advanced(kwargs["contour_padding"], context))) if "contour_padding" in kwargs else 0
        min_area = int(
            float(self.Unprompted.parse_advanced(kwargs["min_area"], context))) if "min_area" in kwargs else 50
        target_mask = self.Unprompted.parse_alt_tags(kwargs["mask"], context) if "mask" in kwargs else "face"
        target_region = self.Unprompted.parse_alt_tags(kwargs["mask_region"], context) if "mask_region" in kwargs else "face"
        instance_mask = self.Unprompted.parse_alt_tags(kwargs["instance"],
                                                       context) if "instance" in kwargs else "person"

        set_pargs = pargs
        set_kwargs = kwargs
        set_pargs.insert(0, "return_image")

        if context == "after":
            all_images = self.Unprompted.after_processed.images
        else:
            all_images = self.Unprompted.shortcode_user_vars["init_images"]
            
        if img2img:
            all_images = self.Unprompted.shortcode_user_vars["init_images"]

        append_originals = []
        
        def sort_instance_masks_by_centroid(instances_mask, reverse=False):
            # Calculate centroid of each instance mask
            centroids = []
            for mask in instances_mask:
                # Convert mask to numpy array
                mask_np = numpy.array(mask)

                # Find contours of the mask
                contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Calculate moments of the contour
                moments = cv2.moments(contours[0])

                # Calculate centroid coordinates
                c_x = int(moments["m10"] / moments["m00"])
                c_y = int(moments["m01"] / moments["m00"])
                centroids.append((c_x, c_y))

            # Sort instance masks by centroid coordinates
            sorted_instances_mask = [instance_mask for _, instance_mask in
                                     sorted(zip(centroids, instances_mask), reverse=reverse)]

            return sorted_instances_mask

        

        def sort_instance_masks(instances_mask,reverse=False,dialate=False):
    
            # 计算每个实例掩码的边界框中心点的水平坐标
            centers = []
            for mask in instances_mask:
                # 将PIL Image转换为numpy数组
                mask_np = numpy.array(mask)

                # 计算掩码的轮廓
                contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # 计算轮廓的边界框
                x, y, w, h = cv2.boundingRect(contours[0])

                # 计算边界框的中心点的水平坐标
                center_x = x + w / 2
                centers.append(center_x)

            # 按照中心点的水平坐标排序实例掩码
            sorted_instances_mask = [instance_mask for _, instance_mask in sorted(zip(centers, instances_mask),reverse=reverse)]

            return sorted_instances_mask

        sort_method = sort_instance_masks
        if sort_by_centroid:
            sort_method = sort_instance_masks_by_centroid
        # Batch support yo
        for image_idx, image_pil in enumerate(all_images):
            self.Unprompted.log("len images is {}".format(len(all_images)))
            instances_masks = []
            if instance_mask != "none":
                self.Unprompted.shortcode_user_vars["init_images"] = [image_pil]
                self.Unprompted.shortcode_objects["instance2mask_opt"].run_block(set_pargs, set_kwargs, None, instance_mask)
                instances_masks = self.Unprompted.shortcode_user_vars["image_masks"]
                #self.Unprompted.shortcode_user_vars["init_images"] = []
         
                if mask_sort_method == "right-to-left":
                    instances_masks = sort_method(instances_masks,True)
                else:
                    instances_masks = sort_method(instances_masks,False)
                
                if save:
                    for idx, ins_img_pil in enumerate(instances_masks):
                        ins_img_pil.save("instance_enhance_instance{}.png".format(idx))

            image_copy = image_pil.copy()
            image = numpy.array(image_pil)
            if save: image_pil.save("instance_enhance_0{}.png".format(image_idx))
                
            image_processed = image_pil.copy()
            if use_workaround:
                append_originals.append(image_processed.copy())

            for c_idx, ins_mask in enumerate(instances_masks):
                if instance_dialate:
                    thresh = cv2.threshold(numpy.array(ins_mask), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                    # Two pass dilate with horizontal and vertical kernel
                    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
                    dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
                    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
                    ins_mask = cv2.dilate(dilate, vertical_kernel, iterations=2)
                    ins_mask = Image.fromarray(ins_mask)
                width, height = image_pil.size
                self.Unprompted.log(f"Processing instance #{c_idx + 1}...")
                # Set prompt with multi-subject support
                self.Unprompted.shortcode_user_vars["prompt"] = all_replacements[min(c_idx, len(all_replacements) - 1)]
             
            
                #image_copy = Image.composite(image_copy, back_ground,
                #                             PIL.ImageOps.invert(ins_mask.convert('L')).filter(PIL.ImageFilter.MaxFilter(3)))
            
                blur_radius = math.ceil(width * blur_radius_orig) // 2 * 2 + 1
                #if blur_radius > 0:
                #    ins_mask = ins_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

                #print(numpy.array(PIL.ImageOps.invert(ins_mask)))
                init_image_with_masg = {"image": image_processed.convert('RGBA'), "mask": ins_mask.convert('RGBA')}
                self.Unprompted.shortcode_user_vars["mode"] = 2
                self.Unprompted.shortcode_user_vars["init_img_with_mask"] = init_image_with_masg
                self.Unprompted.shortcode_user_vars["width"] = width
                self.Unprompted.shortcode_user_vars["height"] = height
                self.Unprompted.shortcode_user_vars["inpainting_fill"] = inpainting_fill
                self.Unprompted.shortcode_user_vars["inpaint_full_res"] = False
                

                fixed_image = self.Unprompted.shortcode_objects["img2img_opt"].run_atomic(set_pargs, None, None)
                if not fixed_image:
                    self.Unprompted.log("An error occurred! Perhaps the user interrupted the operation?")
                    return ""
                
                image_processed = fixed_image
                if save:
                    image_processed.save("image_processed{}.png".format(c_idx))

                self.Unprompted.log("fixed_image size:{}:{}".format(*list(fixed_image.size)))
                self.Unprompted.log("image_copy size:{}:{}".format(*list(image_copy.size)))
                
                # test outside after block, WIP pls don't use
                if context != "after":
                    self.Unprompted.shortcode_user_vars["init_images"] = image_pil

                # Remove temp image key in case [img2img] is used later
                if "img2img_init_image" in self.Unprompted.shortcode_user_vars: del self.Unprompted.shortcode_user_vars[
                    "img2img_init_image"]
            
            if ema_face:
                set_kwargs["txt2mask_init_image"] = image_copy
                self.Unprompted.shortcode_user_vars["mode"] = 0
                self.Unprompted.shortcode_user_vars["image_mask"] = None
                set_kwargs["save"] = "txtmask_ema_face{}".format(image_idx)
                #image = numpy.array(image_processed)
                ema_mask = self.Unprompted.shortcode_objects["txt2mask"].run_block(set_pargs, set_kwargs, None,
                                                                                     target_region)
                ema_mask = cv2.cvtColor(numpy.array(ema_mask), cv2.COLOR_BGR2GRAY)
                if ema_face_dialate:
                    thresh = cv2.threshold(ema_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                    # Two pass dilate with horizontal and vertical kernel
                    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
                    dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
                    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
                    ema_mask = cv2.dilate(dilate, vertical_kernel, iterations=2)
                ema_mask = Image.fromarray(ema_mask)
                if save:
                    ema_mask.save("ema_mask.png")
                
                        
            if target_mask == "face_extra":
                if use_workaround:
                    append_originals.append(image_processed.copy())
                set_kwargs["txt2mask_init_image"] = image_processed
                self.Unprompted.shortcode_user_vars["mode"] = 0
                self.Unprompted.shortcode_user_vars["image_mask"] = ema_mask if ema_face else None
                set_kwargs["save"] = "txtmask{}".format(image_idx)
                image = numpy.array(image_processed)
                mask_image = self.Unprompted.shortcode_objects["txt2mask"].run_block(set_pargs, set_kwargs, None,
                                                                                     target_region)
                mask_image = cv2.cvtColor(numpy.array(mask_image), cv2.COLOR_BGR2GRAY)
                if save:
                    Image.fromarray(mask_image).save("extra_face_mask.png")

                thresh = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # Two pass dilate with horizontal and vertical kernel
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
                dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
                dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)

                # Find contours, filter using contour threshold area
                cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                if mask_sort_method is not "unsorted":
                    if mask_sort_method=="small-to-big":
                        cnts = sorted(cnts, key=cv2.contourArea, reverse=False)[:5]
                    elif mask_sort_method=="big-to-small":
                        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
                    else:
                        # initialize the reverse flag and sort index
                        reverse = False
                        i = 0
                        # handle if we need to sort in reverse
                        if mask_sort_method == "right-to-left" or mask_sort_method == "bottom-to-top": reverse = True
                        # handle if we are sorting against the y-coordinate rather than
                        # the x-coordinate of the bounding box
                        if mask_sort_method == "top-to-bottom" or mask_sort_method == "bottom-to-top": i = 1
                        # construct the list of bounding boxes and sort them from top to
                        # bottom
                        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                            key=lambda b:b[1][i], reverse=reverse))

                #i = 0
                #boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                #(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                #                                key=lambda b: b[1][i], reverse=False))
                for e_idx, c in enumerate(cnts):
                    self.Unprompted.shortcode_user_vars["width"] = upscale_width
                    self.Unprompted.shortcode_user_vars["height"] = upscale_height
                    self.Unprompted.log(f"Processing contour face extra#{e_idx + 1}...")
                    area = cv2.contourArea(c)
                    if area >= min_area:
                        x, y, w, h = cv2.boundingRect(c)
                        self.Unprompted.log(f"Contour properties: {x} {y} {w} {h}")

                        # Make sure it's a square, 1:1 AR for stable diffusion
                        size = max(w, h)
                        w = size
                        h = size
                        # Padding may be constrained by the mask region position
                        padding = min(padding_original, x, y)

                        if "denoising_strength" not in kwargs or "cfg_scale" not in kwargs:
                            target_size = (w * h) / (self.Unprompted.shortcode_user_vars["width"] *
                                                 self.Unprompted.shortcode_user_vars["height"])
                            self.Unprompted.log(f"Masked region size is {target_size}")
                            if target_size < target_size_max:
                                sig = sigmoid(-6 + (
                                        target_size / target_size_max) * 12)  # * -1 # (12 * (target_size / target_size_max) - 6))
                                self.Unprompted.log(f"Sigmoid value: {sig}")
                                if "denoising_strength" not in kwargs:
                                    self.Unprompted.shortcode_user_vars["denoising_strength"] = (1 - sig) * denoising_max
                                    self.Unprompted.log(f"Denoising strength is {self.Unprompted.shortcode_user_vars['denoising_strength']}")
                                if "cfg_scale" not in kwargs:
                                    self.Unprompted.shortcode_user_vars["cfg_scale"] = cfg_min + sig * cfg_max
                                    self.Unprompted.log(
                                        f"CFG Scale is {self.Unprompted.shortcode_user_vars['cfg_scale']} (min {cfg_min}, max {cfg_min + cfg_max})")
                            else:
                                self.Unprompted.log("Humongous target detected. Skipping zoom_enhance...")
                                continue

                        # Set prompt with multi-subject support
                        self.Unprompted.shortcode_user_vars["prompt"] = all_replacements[
                            min(e_idx, len(all_replacements) - 1)]
                        #self.Unprompted.shortcode_user_vars["negative_prompt"] = all_negative_replacements[
                        #    min(c_idx, len(all_negative_replacements) - 1)]

                        y1 = max(0, y - padding)
                        y2 = min(image.shape[0], y + h + padding)
                        # In case the target appears close to the bottom of the picture, we push the mask up to get the right 1:1 size
                        if (y2 - y1 < size): y1 -= size - (y2 - y1)

                        x1 = max(0, x - padding)
                        x2 = min(image.shape[1], x + w + padding)
                        if (x2 - x1 < size): x1 -= size - (x2 - x1)

                        sub_img = Image.fromarray(image[y1:y2, x1:x2])
                        sub_mask = Image.fromarray(mask_image[y1:y2, x1:x2])
                        sub_img_big = sub_img.resize((upscale_width, upscale_height), resample=upscale_method)
                        if save:
                            sub_img_big.save("{}faceextrasource.png".format(e_idx))
                        #sub_mask_big = sub_mask.resize((upscale_width, upscale_height), resample=upscale_method)

                        # blur radius is relative to canvas size, should be odd integer
                        blur_radius = math.ceil(w * blur_radius_orig) // 2 * 2 + 1
                        if blur_radius > 0:
                            sub_mask = sub_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

                        self.Unprompted.shortcode_user_vars["mode"] = 0
                        #init_image_with_masg = {"image": sub_img_big.convert('RGBA'), "mask": sub_mask_big.convert('RGBA')}
                        #self.Unprompted.shortcode_user_vars["init_img_with_mask"] = init_image_with_masg
                        self.Unprompted.shortcode_user_vars["img2img_init_image"] = sub_img_big
                        #self.Unprompted.shortcode_user_vars["inpaint_full_res"] = 0
                        #self.Unprompted.shortcode_user_vars["inpainting_fill"] = inpainting_fill
                        
                        fixed_image = self.Unprompted.shortcode_objects["img2img_opt"].run_atomic(set_pargs, None, None)
                        if not fixed_image:
                            self.Unprompted.log("An error occurred! Perhaps the user interrupted the operation?")
                            continue
                        # self.Unprompted.shortcode_user_vars["init_images"].append(fixed_image)
                        # Downscale fixed image back to original size
                        fixed_image = fixed_image.resize((w + padding * 2, h + padding * 2), resample=downscale_method)

                        # Slap our new image back onto the original
                        image_processed.paste(fixed_image,(x1 - padding, y1 - padding), sub_mask)
                        
                        if save:
                            image_pil.save("{}image_processed_2.png".format(e_idx))

                        # test outside after block, WIP pls don't use
                        if context != "after":
                            self.Unprompted.shortcode_user_vars["init_images"] = image_processed

                        # Remove temp image key in case [img2img] is used later
                        if "img2img_init_image" in self.Unprompted.shortcode_user_vars: del \
                            self.Unprompted.shortcode_user_vars["img2img_init_image"]
            
            if image_processed:
                append_originals.append(image_processed.copy())

        # Add original images
        for appended_image in append_originals:
            self.Unprompted.after_processed.images.append(appended_image)
        #self.Unprompted.after_processed.images = append_originals

        self.Unprompted.shortcode_user_vars["batch_size"] = batch_size_orig
        self.Unprompted.shortcode_user_vars["n_iter"] = n_iter_orig

        return ""


def ui(self, gr):
    gr.Checkbox(label="Final image not showing up? Try using this workaround 🡢 use_workaround")
    gr.Text(label="Mask to find 🡢 mask", value="face")
    gr.Text(label="Replacement 🡢 replacement", value="face")
    gr.Text(label="Negative replacement 🡢 negative_replacement", value="")
    gr.Dropdown(label="Mask sorting method 🡢 mask_sort_method", value="left-to-right",
                choices=["left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top", "big-to-small",
                         "small-to-big", "unsorted"])
    gr.Checkbox(label="Include original image in output window 🡢 include_original")
    gr.Checkbox(label="Save debug images to WebUI folder 🡢 save")
    gr.Checkbox(label="Unload txt2mask model after inference (for low memory devices) 🡢 unload_model")
    with gr.Accordion("⚙️ Advanced Options", open=False):
        gr.Dropdown(label="Upscale method 🡢 upscale_method", value="Nearest Neighbor",
                    choices=list(self.resample_methods.keys()), interactive=True)
        gr.Dropdown(label="Downscale method 🡢 downscale_method", value="Lanczos",
                    choices=list(self.resample_methods.keys()), interactive=True)
        gr.Slider(label="Blur edges size 🡢 blur_size", value=0.03, maximum=1.0, minimum=0.0, interactive=True,
                  step=0.01)
        gr.Slider(label="Minimum CFG scale 🡢 cfg_scale_min", value=3.0, maximum=15.0, minimum=0.0, interactive=True,
                  step=0.5)
        gr.Slider(label="Maximum denoising strength 🡢 denoising_max", value=0.65, maximum=1.0, minimum=0.0,
                  interactive=True, step=0.01)
        gr.Slider(label="Maximum mask size (if a bigger mask is found, it will bypass the shortcode) 🡢 mask_size_max",
                  value=0.3, maximum=1.0, minimum=0.0, interactive=True, step=0.01)
        gr.Text(label="Force denoising strength to this value 🡢 denoising_strength")
        gr.Text(label="Force CFG scale to this value 🡢 cfg_scale")
        gr.Number(label="Mask minimum number of pixels 🡢 min_area", value=50, interactive=True)
        gr.Number(label="Contour padding in pixels 🡢 contour_padding", value=0, interactive=True)
        gr.Number(label="Upscale width 🡢 upscale_width", value=512, interactive=True)
        gr.Number(label="Upscale height 🡢 upscale_height", value=512, interactive=True)
