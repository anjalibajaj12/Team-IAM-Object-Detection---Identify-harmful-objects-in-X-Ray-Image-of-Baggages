# we have taken help from the following links:
# https://debuggercafe.com/object-detection-using-ssd300-resnet50-and-pytorch/
# https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/


#import torch
import torch
#import opencv
import cv2
import torchvision.transforms as transforms
# we added the imshow library as there is some issue going with google colab using cv2.imshow
from google.colab.patches import cv2_imshow

# draw the bounding boxes around the objects in an image


def draw_bboxes(image, results, classes_to_labels):
    for image_idx in range(len(results)):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = image.shape[0], image.shape[1]
        bboxes, classes, confidences = results[image_idx]
        for idx in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[idx]
            x1, y1 = int(x1*300), int(y1*300)
            x2, y2 = int(x2*300), int(y2*300)
            x1, y1 = int((x1/300)*orig_w), int((y1/300)*orig_h)
            x2, y2 = int((x2/300)*orig_w), int((y2/300)*orig_h)
            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                image, classes_to_labels[classes[idx]-1], (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

    return image


# for enabling gpu for running and executing the code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# loading ssd-resnet model taken help from ssd pytorch ssd implementation
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                           'nvidia_ssd', map_location=torch.device('cpu'))
ssd_model.to(device)
ssd_model.eval()
utils = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')


# reading the image path and converting it to the tensor
image_path = "/content/x-ray-image-showing-briefcase-containing-knife-CTF3RC.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
transformed_image = transform(image)
tensor = torch.tensor(transformed_image, dtype=torch.float32)
tensor = tensor.unsqueeze(0).to(device)

# passing the tensor through the ssd resnet model and obtaining the output images with the labels.
with torch.no_grad():
    detections = ssd_model(tensor)
results_per_input = utils.decode_results(detections)
best_results_per_input = [utils.pick_best(
    results, 0.45) for results in results_per_input]
classes_to_labels = utils.get_coco_object_dictionary()
image_result = draw_bboxes(image, best_results_per_input, classes_to_labels)
cv2_imshow(image_result)
cv2.waitKey(0)
save_name = image_path.split('/')[-1]
cv2.imwrite(f"outputs/{save_name}", image_result)
