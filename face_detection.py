from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from PIL import Image

def draw_image_with_boxes(pixels, result_list):
    # plot the image
    pyplot.imshow(pixels)

    # get the context for drawing boxes
    ax = pyplot.gca()

    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']

        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')

        # draw the box
        ax.add_patch(rect)

    # show the plot
    pyplot.show()
    
    # get face coordinates
    return(x,y,x+width,y+height)

if __name__ == "__main__":
    filename = './dataset/sample/real/DSC_0242.jpg'

    detector = MTCNN()

    # Reading Image File
    pixels = pyplot.imread(filename)

    # MTCNN Face Detection
    faces = detector.detect_faces(pixels)

    # display faces on the original image
    draw_image_with_boxes(pixels, faces)