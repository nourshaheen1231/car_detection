import cv2 as cv
import numpy as np
from collections import deque, Counter
import MNN


def resize_and_pad(img, size=(224, 224), pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size

    interp = cv.INTER_AREA if (h > sh or w > sw) else cv.INTER_CUBIC
    aspect = w / h

    if aspect > 1:  # Horizontal image
        new_w = sw
        new_h = int(np.round(new_w / aspect))
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = int(np.floor(pad_vert)), int(np.ceil(pad_vert))
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # Vertical image
        new_h = sh
        new_w = int(np.round(new_h * aspect))
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = int(np.floor(pad_horz)), int(np.ceil(pad_horz))
        pad_top, pad_bot = 0, 0
    else:  # Square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color] * 3

    scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
    return cv.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv.BORDER_CONSTANT,
        value=pad_color,
    )


class CarMakeModelDetection:

    def __init__(
        self,
        model_path: str,
        class_names: list,
        history_size: int = 15,
        input_size: tuple = (224, 224),
        min_confidence: float = 0.4,
        compute_interval: int = 5,
    ):
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)

        self.class_names = class_names
        self.input_size = input_size
        self.min_confidence = min_confidence
        self.history_size = history_size
        self.compute_interval = compute_interval

        self._history = {}
        self._cache = {}
        self._frame_counter = 0

    def get_stable_make_model(self, track_id, frame, bbox):
        self._frame_counter += 1

        if track_id in self._cache:
            if self._frame_counter % self.compute_interval != 0:
                return self._cache[track_id]["make_model"]

        make_model, confidence = self._predict_make_model(frame, bbox)

        if make_model != "Unknown":
            self._cache[track_id] = {
                "make_model": make_model,
                "confidence": confidence,
            }

        print(
            f"[Car {track_id}] Make/Model: {make_model} | Confidence: {confidence:.2f}"
        )

        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.history_size)

        if make_model != "Unknown":
            self._history[track_id].append(make_model)

        if not self._history[track_id]:
            return "Unknown"

        final_make_model = Counter(self._history[track_id]).most_common(1)[0][0]
        self._cache[track_id]["make_model"] = final_make_model

        return final_make_model

    def _predict_make_model(self, frame, bbox):
        crop = self._crop(frame, bbox)
        if crop is None:
            return "Unknown", 0.0

        # 1. BGR -> RGB (matching Spectrico img[:, :, ::-1])
        crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)

        # 2. Aspect-ratio padding resize
        crop_padded = resize_and_pad(crop_rgb, self.input_size)

        # 3. Scaling to [-1.0, 1.0] (matching Spectrico img / 127.5 - 1.0)
        img_data = crop_padded.astype(np.float32) / 127.5 - 1.0

        # 4. NHWC -> NCHW conversion for MNN Caffe shape convention
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        tmp_input = MNN.Tensor(
            (1, 3, self.input_size[1], self.input_size[0]),
            MNN.Halide_Type_Float,
            np.ascontiguousarray(img_data),
            MNN.Tensor_DimensionType_Caffe,
        )

        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)

        output_tensor = self.interpreter.getSessionOutput(self.session)
        output_host = MNN.Tensor(output_tensor, MNN.Tensor_DimensionType_Caffe)
        output_tensor.copyToHostTensor(output_host)

        preds = np.array(output_host.getData())

        idx = int(np.argmax(preds))
        conf = float(preds[idx])

        if conf < self.min_confidence or idx >= len(self.class_names):
            return "Unknown", conf

        return self.class_names[idx], conf

    def _crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(x1, 0), max(y1, 0)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def reset(self):
        self._history.clear()
        self._cache.clear()
        self._frame_counter = 0