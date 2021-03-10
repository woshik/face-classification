from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from PIL import Image
import os

path = './dataset/'
folders = os.listdir(path)
detector = MTCNN()

def cropFaceAndSaveImage(fileName, destination):
    try:
        # Reading Image File
        pixels = pyplot.imread(fileName)

        # MTCNN Face Detection
        faces = detector.detect_faces(pixels)

        # Getting Co-ordinates of the faces
        x, y, width, height = faces[0]['box']

        # Crop and Save the image
        Image.fromarray(pixels).crop(
            (x, y, x+width, y+height)).save(destination)

    except (IndexError or SystemError):
        print('Face Not Found')


def processImages(folder):
    # Iterate over the folder
    for subs in folders:
        for fileName in os.listdir(path+subs):
            if 'Fake' in path+subs+fileName and 'jpg' in path+subs+fileName:
                location = path+subs+'/'+fileName
                destination = 'training/fake/'+fileName

                print(location)

                cropFaceAndSaveImage(location, destination)

            elif 'Real' in path+subs+fileName and 'jpg' in path+subs+fileName:
                location = path+subs+'/'+fileName
                destination = 'training/real/'+fileName

                print(location)

                cropFaceAndSaveImage(location, destination)


if __name__ == "__main__":
    processImages(folders)
