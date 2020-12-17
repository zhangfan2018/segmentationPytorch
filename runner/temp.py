def test_frac_patch_seg(self, test_dataset):
    """test procedure."""
    self.network.eval()
    self.network = self.network.half()

    csv_path = os.path.join(self.args.out_dir, 'infer_result.csv')
    contents = ['uid', 'coordX', 'coordY', 'coordZ',
                'detector_diameterX', 'detector_diameterY', 'detector_diameterZ',
                'boneNo', 'boneType', 'frac_type', 'det_probability', 'candidate_type']
    write_csv(csv_path, contents, mul=False, mod="w")

    self.logger.info('starting test')
    self.logger.info('the number of test dataset: {}'.format(len(test_dataset)))

    for data_dict in tqdm(test_dataset):
        uid = data_dict["uid"]
        candidates = data_dict["candidates"]
        image_shape = data_dict["image_shape"]
        zoom_factor = data_dict["zoom_factor"]
        batch_num = 120
        images = []
        output_seg = []
        for index, candidate in enumerate(candidates):
            if index % batch_num == 0:
                images = []
            image = candidate["image"]
            images.append(image)

            if len(images) == batch_num or index == len(candidates) - 1:
                images = np.stack(images, axis=0)
                images = torch.from_numpy(images).float().half()
                if self.args.cuda:
                    images = images.cuda()

                with torch.no_grad():
                    batch_output = self.network(images)
                output_seg.append(batch_output)

        output_seg = torch.cat(output_seg, 0)
        output_seg = F.sigmoid(output_seg)
        output_seg = output_seg.cpu().numpy().squeeze()
        output_seg[output_seg >= 0.5] = 1
        output_seg[output_seg < 0.5] = 0

        mask = np.zeros(image_shape, np.uint8)
        for i in range(len(candidates)):
            out_candidate_seg = output_seg[i, ...].squeeze()
            crop_bbox = candidates[i]["crop_bbox"]
            part_mask_ori = mask[crop_bbox[0]:crop_bbox[1],
                            crop_bbox[2]:crop_bbox[3],
                            crop_bbox[4]:crop_bbox[5]]
            part_mask_dst = out_candidate_seg[:crop_bbox[1] - crop_bbox[0],
                            :crop_bbox[3] - crop_bbox[2],
                            :crop_bbox[5] - crop_bbox[4]]
            part_mask_ori[part_mask_dst != 0] = 1
            mask[crop_bbox[0]:crop_bbox[1],
            crop_bbox[2]:crop_bbox[3],
            crop_bbox[4]:crop_bbox[5]] = part_mask_ori

        all_candidates = extract_candidates_bbox(mask, area_least=10)
        for candidate in all_candidates:
            centroid = candidate["centroid"]
            bbox = candidate["bbox"]
            raw_centroid = [centroid[i] * zoom_factor[i] for i in range(3)]
            raw_diameter = [(bbox[3] - bbox[0]) * zoom_factor[0],
                            (bbox[4] - bbox[1]) * zoom_factor[1],
                            (bbox[5] - bbox[2]) * zoom_factor[2]]
            contents = [uid, raw_centroid[2], raw_centroid[1], raw_centroid[0],
                        raw_diameter[2], raw_diameter[1], raw_diameter[0],
                        -1, -1, -1, 1, -1]
            write_csv(csv_path, contents, mul=False, mod="a+")