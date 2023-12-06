# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

"""## 사용 예

### 이미지 다운로드 및 시각화를 위한 도우미 함수

가장 간단한 필수 기능을 제공하도록 [TF 물체 감지 API](https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py)에서 조정된 시각화 코드입니다.
"""


# 비율로 저거 저장하는거
def crop_image_relative(input_path, output_path, top, left, bottom, right):
    # 이미지 열기
    original_image = Image.open(input_path)

    # 이미지 크기 얻기
    width, height = original_image.size

    # 상대적인 좌표를 실제 좌표로 변환
    left_pixel = int(left * width)
    top_pixel = int(top * height)
    right_pixel = int(right * width)
    bottom_pixel = int(bottom * height)

    # 이미지 자르기
    cropped_image = original_image.crop((left_pixel, top_pixel, right_pixel, bottom_pixel))

    # 추출된 부분 저장
    cropped_image.save(output_path)


def image_save(input_image_path, output_name, detection_box, index):
    #output_image_path = f"/Users/dejeong/Desktop/img_test/{output_name}.png"  # 파일 경로를 실제 파일 경로로 변경하세요
    output_image_path = f"/Users/dejeong/Desktop/Study/Hack2/hack2/public/{output_name}.png"
    # 추출할 부분의 좌표 (top, left, bottom, right), 모두 0과 1 사이의 상대적인 값

    locations = detection_box
    top_value, left_value, bottom_value, right_value = locations

    # 이미지 자르기 및 저장
    crop_image_relative(input_image_path, output_image_path, top_value, left_value, bottom_value, right_value)


def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    plt.savefig('savefig_default.png')


def download_and_resize_image(url, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                    fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                display_str,
                fill="black",
                font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                           int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


"""## 모듈 적용하기

Open Images v4에서 공개 이미지를 로드하고 로컬에 저장한 다음 표시합니다.
"""

# By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
# image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"  #@param

image_url = "https://images.chosun.com/resizer/8zcAsADeBxO9L2tESlocaGTnKc4=/530x988/smart/cloudfront-ap-northeast-1.images.arcpublishing.com/chosun/T3ZWMNANM732IPWSNBEJGSQEQU.jpg"
downloaded_image_path = download_and_resize_image(image_url, 480, 878, True)

"""물체 감지 모듈을 선택하고 다운로드한 이미지에 적용합니다. 모듈:

- **FasterRCNN+InceptionResNet V2**: 높은 정확성
- **ssd + mobilenet V2**: 작고 빠름
"""

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"  # @param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]

detector = hub.load(module_handle).signatures['default']


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(detector, path):
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)

    image_with_boxes = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])
    print("========================")
    print("result : ", result)
    print("========================")

    display_image(image_with_boxes)


    for i in range(0, 5):
        now_name = result['detection_class_entities'][i]
        now_detection_box = result['detection_boxes'][i]
        image_save(downloaded_image_path, now_name, now_detection_box, i)

run_detector(detector, downloaded_image_path)
