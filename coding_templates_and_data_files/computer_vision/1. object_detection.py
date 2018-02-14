# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
def detect(frame, net, transform):
    """
    Frame = the image
    net = the network
    transform = transformations applied to the image. Making the images compatible with the network
    """
    # Getting the height and width of the image
    height, width = frame.shape[:2]
    # Tranformed frame after transformation
    frame_t = transform(frame)[0]
    # NumPy array to Torch Tensor
    # Invert red, blue, green to green, red, blue (was trained on these colours in that order)
    # Permute does this, 2 being green, 0 being red and 1 being blue
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    # Add new dimension (unsqueeze), should always be first index - NN is only able to accept batches of data (Pytorch)
    # Variable - torch variable that contains a tensor and a gradient
    x = Variable(x.unsqueeze(0))
    # Feeds x to our neural network
    y = net(x)
    # Retrieve the values of output
    detections = y.data
    # First width & height = top left corner
    # Second width & height = bottom right corner
    # Used to normalize the scaled values between 0 & 1
    scale = torch.Tensor([width, height, width, height])
    
    # The detections Tensor consists of:
    # [batch, number of classes/objects, number of occurence, (score, x0, y0, x1, y1)]
    
    # Number of classes = detections.size()
    for i in range(detections.size(1)):
        # Occurence of the objects
        j = 0
        # While the score (last 0) of the occurence (j) is detected (i) is greater than 0.6, continue loop
        # first 0 = batch
        while detections[0, i, j, 0] >= 0.6:
            # Point (pt) is a torch Tensor
            # 1: = last 4 elements in the tuple mentioned above within the Tensor (x0, y0, x1, y1)
            pt = (detections[0, i, j, 1:] * scale).numpy()
            # pt[0] -> pt[3] = (x0, y0, x1, y1)
            # Creates the rectangle
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            # Creates the label
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # Update the occurence of the objects
            j += 1
    return frame

# Creating the SSD neural network
net = build_ssd('test')
# torch.load = create weights for the network. Using an anonymous function we create a function called storage
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

# Creating the transformation
# net.size = target size of the images to be given to the NN
# the values in the tuple are scale values to ensure that the colour values are in the correct scale
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader('epic-horses.mp4')
# Outputs the fps
fps = reader.get_meta_data()['fps']
# Create the output video
writer = imageio.get_writer('horse_output.mp4', fps = fps)
for i, frame in enumerate(reader):
    # .eval() - allows Python to run code within itself, meaning it grabs 'y' instead of 'build_ssd'
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()