import cv2
import numpy as np

from timm.models.layers import to_2tuple
import torch

from utils import pad_image_to_shape, normalize


class Evaluator(object):
    def __init__(self, class_num, norm_mean1, norm_std1, norm_mean2, norm_std2, network, multi_scales, is_flip):
        self.class_num = class_num
        self.norm_mean = np.array(norm_mean1)
        self.norm_std = np.array(norm_std1)
        self.norm_mean_X = np.array(norm_mean2)
        self.norm_std_X = np.array(norm_std2)
        self.multi_scales = multi_scales
        self.is_flip = is_flip

        self.val_func = network

    # evaluate the whole image at once
    def whole_eval(self, img, output_size, device=None):
        processed_pred = np.zeros(
            (output_size[0], output_size[1], self.class_num))

        for s in self.multi_scales:
            scaled_img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            scaled_img = self.process_image(scaled_img, None)
            pred = self.val_func_process(scaled_img, device)
            pred = pred.permute(1, 2, 0)
            processed_pred += cv2.resize(pred.cpu().numpy(),
                                         (output_size[1], output_size[0]),
                                         interpolation=cv2.INTER_LINEAR)

        pred = processed_pred.argmax(2)

        return pred

    # slide the window to evaluate the image
    def sliding_eval(self, img, crop_size, stride_rate, device=None):
        ori_rows, ori_cols, c = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process(img_scale,
                                                 (ori_rows, ori_cols),
                                                 crop_size, stride_rate, device)

        pred = processed_pred.argmax(2)

        return pred

    def scale_process(self, img, ori_shape, crop_size, stride_rate,
                      device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, margin = self.process_image(img, crop_size)
            score = self.val_func_process(input_data, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score = self.val_func_process(input_data, device)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            # score = data_scale / count_scale
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process(self, input_data, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                          dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data)
                score = torch.squeeze(score[0], dim=0)

                if self.is_flip:
                    input_data = input_data.flip(-1)
                    input_data = input_data.flip(-2)
                    score_flip = self.val_func(input_data)
                    score_flip = torch.squeeze(score_flip[0], dim=0)
                    score_flip = score_flip.flip(-2)
                    score += score_flip.flip(-1)
                # score = torch.exp(score)
                # score = score.data

        return score

    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.norm_mean, self.norm_std)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img

    # evaluate the whole image at once
    def whole_eval_rgbX(self, img, modal_x, device=None):
        processed_pred = np.zeros(
            (img.shape[0], img.shape[1], self.class_num))

        for s in self.multi_scales:
            scaled_img = cv2.resize(img, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_LINEAR)
            modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

            input_data, input_modal_x = self.process_image_rgbX(scaled_img, modal_x_scale, None)
            pred = self.val_func_process_rgbX(input_data, input_modal_x, device)

            pred = pred.permute(1, 2, 0)
            processed_pred += cv2.resize(pred.cpu().numpy(),
                                         (img.shape[1], img.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)

        pred = processed_pred.argmax(2)

        return pred

    # add new funtion for rgb and modal X segmentation
    def sliding_eval_rgbX(self, img, modal_x, crop_size, stride_rate, device=None):
        crop_size = to_2tuple(crop_size)
        ori_rows, ori_cols, _ = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process_rgbX(img_scale, modal_x_scale, (ori_rows, ori_cols),
                                                      crop_size, stride_rate, device)

        pred = processed_pred.argmax(2)

        return pred

    def scale_process_rgbX(self, img, modal_x, ori_shape, crop_size, stride_rate, device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
            input_data, input_modal_x, margin = self.process_image_rgbX(img, modal_x, crop_size)
            score = self.val_func_process_rgbX(input_data, input_modal_x, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = (int(np.ceil(crop_size[0] * stride_rate)), int(np.ceil(crop_size[1] * stride_rate)))
            img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
            modal_x_pad, margin = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride[0]
                    s_y = grid_yidx * stride[1]
                    e_x = min(s_x + crop_size[0], pad_cols)
                    e_y = min(s_y + crop_size[1], pad_rows)
                    s_x = e_x - crop_size[0]
                    s_y = e_y - crop_size[1]
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    if len(modal_x_pad.shape) == 2:
                        modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x]
                    else:
                        modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x, :]

                    input_data, input_modal_x, tmargin = self.process_image_rgbX(img_sub, modal_x_sub, crop_size)
                    temp_score = self.val_func_process_rgbX(input_data, input_modal_x, device)

                    temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process_rgbX(self, input_data, input_modal_x, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)

        input_modal_x = np.ascontiguousarray(input_modal_x[None, :, :, :], dtype=np.float32)
        input_modal_x = torch.FloatTensor(input_modal_x).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data, input_modal_x)
                score = torch.squeeze(score[0], dim=0)
                if self.is_flip:
                    input_data = input_data.flip(-1)
                    input_modal_x = input_modal_x.flip(-1)
                    input_data = input_data.flip(-2)
                    input_modal_x = input_modal_x.flip(-2)
                    score_flip = self.val_func(input_data, input_modal_x)
                    score_flip = torch.squeeze(score_flip[0], dim=0)
                    score_flip = score_flip.flip(-2)
                    score += score_flip.flip(-1)
                score = torch.exp(score)

        return score

    # for rgbd segmentation
    def process_image_rgbX(self, img, modal_x, crop_size=None):
        p_img = img
        p_modal_x = modal_x

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), amodal_xis=2)

        p_img = normalize(p_img, self.norm_mean, self.norm_std)
        p_modal_x = normalize(p_modal_x, self.norm_mean_X, self.norm_std_X)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_modal_x, _ = pad_image_to_shape(p_modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)
            if len(modal_x.shape) == 2:
                p_modal_x = p_modal_x[np.newaxis, ...]
            else:
                p_modal_x = p_modal_x.transpose(2, 0, 1)  # 3 H W

            return p_img, p_modal_x, margin

        p_img = p_img.transpose(2, 0, 1)  # 3 H W

        if len(modal_x.shape) == 2:
            p_modal_x = p_modal_x[np.newaxis, ...]
        else:
            p_modal_x = p_modal_x.transpose(2, 0, 1)

        return p_img, p_modal_x
