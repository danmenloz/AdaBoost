from src.integral_image import IntegralImage as integral
from src.integral_image import get_sum
from enum import Enum
from PIL import Image, ImageDraw
from numpy import inf


class featureType(Enum):
    # featureType .value to access the tuple
    # tuple(width, height)
    TWO_VERTICAL = (1, 2)
    TWO_HORIZONTAL = (2, 1)
    THREE_VERTICAL = (1, 3)
    THREE_HORIZONTAL = (3, 1)
    FOUR = (2, 2)


class HaarLikeFeature(object):
    # class for getting harr-like features
    # how to select the weak learner (the single rectangle feature selection)
    # h_j(x) = 1 if p_j*f_j(x) < p_j*threshold
    # h_j(x) = 0 otherwise

    def __init__(self, feature_type, position, width, height, error=inf, threshold=0, polarity=1, weight=1):
        '''
        : para feature_type: 5 enumerate types in total
        : para position: position of sub-window (top-left)
        : para width, height: size of the image
        '''
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0]+width, position[1]+height)
        self.width = width
        self.height = height
        self.error = error # min observed error during training
        self.threshold = threshold # threshold = 0 (no misclassified images)
        self.polarity = polarity
        self.weight = weight # alpha value used as weight in the strong classifier
        self.scores = None # list of scores on the dataset


    # @property
    # def error(self):
    #     return self.__error

    # @error.setter
    # def error(self, value):
    #     self.__error = value

    # @property
    # def threshold(self):
    #     return self.__threshold
    
    # @threshold.setter
    # def threshold(self, value):
    #     self.__threshold = value
        
    # @property
    # def polarity(self):
    #     return self.__polarity
    
    # @polarity.setter
    # def polarity(self, value):
    #     self.__polarity = value

    def calc_score(self, int_img):
        score, white, grey = 0, 0, 0

        if self.type == featureType.TWO_VERTICAL:
            white += get_sum(int_img, self.top_left, 
                (int(self.top_left[0] + self.width), int(self.top_left[1] + self.height / 2)))
            grey += get_sum(int_img, (self.top_left[0], 
                int(self.top_left[1] + self.height / 2)), self.bottom_right)

        elif self.type == featureType.TWO_HORIZONTAL:
            white += get_sum(int_img, self.top_left,
                (int(self.top_left[0] + self.width/2), self.top_left[1] + self.height))
            grey += get_sum(int_img,
                (int(self.top_left[0] + self.width/2), self.top_left[1]), self.bottom_right)
            
        elif self.type == featureType.THREE_VERTICAL:
            white += get_sum(int_img, self.top_left,
                (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            grey += get_sum(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)), 
                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            white += get_sum(int_img, (self.top_left[0],
                int(self.top_left[1] + 2 * self.height / 3)), self.bottom_right)

        elif self.type == featureType.THREE_HORIZONTAL:
            white += get_sum(int_img, self.top_left,
                (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            grey += get_sum(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            white += get_sum(int_img, (self.top_left[0], int(
                self.top_left[1] + 2 * self.height / 3)), self.bottom_right)

        elif self.type == featureType.FOUR:
            white += get_sum(int_img, self.top_left,
                (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            grey += get_sum(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            grey += get_sum(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            white += get_sum(int_img, (int(self.top_left[0] + self.width / 2),
                int(self.top_left[1] + self.height / 2)), self.bottom_right)
            
        score = white - grey
        return score

    def get_vote(self, img_idx=None, int_img=None):
        # Get the vote of this feature for a given integral image index
        # note h_j(x) = 0 otherwise
        if int_img is not None:
            score = self.calc_score(int_img) 
        else:
            # scores have already been computed and stored in .scores attribute
            score = self.scores[img_idx]

        if self.polarity == 1:
            vote = 1 if score < self.threshold else 0
        else: # self.polarity = -1
            vote = 1 if score > self.threshold else 0
            
        return vote

    
    def draw_feature(self, img=None, res=None, verbose=0):
        # Draw the feature on a given image or on an empty square
        # img:          PIL image
        # resolution:   (width, height) tuple
        if img is None:
            img = Image.new("RGBA", res, (255, 0, 0, 0))
            # the 4th value in the color parameter (in this example (255, 0, 0, 0)) is set to 0, 
            # the image will be completely transparent - this is the value of the alpha channel
        else:
            img = img.convert('RGBA')
        
        imgr = ImageDraw.Draw( img )
        neg, pos = 'red', 'green' # fill colors
        rectangles = list() # list to hold the rectangles info

        if self.type == featureType.TWO_VERTICAL:
            rect1 = [ self.top_left, 
                (int(self.top_left[0] + self.width), int(self.top_left[1] + self.height / 2)),
                pos ]
            rect2 = [ (self.top_left[0], int(self.top_left[1] + self.height / 2)), 
                self.bottom_right,
                neg ]
            rectangles.extend([rect1, rect2])

        elif self.type == featureType.TWO_HORIZONTAL:
            rect1 = [ self.top_left,
                (int(self.top_left[0] + self.width/2), self.top_left[1] + self.height),
                pos ]
            rect2 = [ (int(self.top_left[0] + self.width/2), self.top_left[1]), 
                self.bottom_right,
                neg ]
            rectangles.extend([rect1, rect2])
            
        elif self.type == featureType.THREE_VERTICAL:
            rect1 = [ self.top_left,
                (self.bottom_right[0], int(self.top_left[1] + self.height / 3)),
                pos ]
            rect2 = [ (self.top_left[0], int(self.top_left[1] + self.height / 3)), 
                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)),
                neg ]
            rect3 = [ (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)), 
                self.bottom_right,
                pos ]
            rectangles.extend([rect1, rect2, rect3])

        elif self.type == featureType.THREE_HORIZONTAL:
            rect1 = [ self.top_left,
                (self.bottom_right[0], int(self.top_left[1] + self.height / 3)),
                pos ]
            rect2 = [ (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)),
                neg ]
            rect3 = [ (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)), 
                self.bottom_right,
                pos ]
            rectangles.extend([rect1, rect2, rect3])

        elif self.type == featureType.FOUR:
            rect1 = [ self.top_left,
                (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)),
                pos ]
            rect2 = [ (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                (self.bottom_right[0], int(self.top_left[1] + self.height / 2)),
                neg ]
            rect3 = [ (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                (int(self.top_left[0] + self.width / 2), self.bottom_right[1]),
                neg ]
            rect4 = [ (int(self.top_left[0] + self.width / 2),
                int(self.top_left[1] + self.height / 2)), self.bottom_right,
                pos ]
            rectangles.extend([rect1, rect2, rect3, rect4])
            
        for rect in rectangles:
            p0, p1, color = rect # extract info
            p1 = (p1[0]-1 ,p1[1]-1) # ignore border 
            imgr.rectangle( [p0, p1], fill = color)
            if verbose:
                print('p0: {}  p1:{}  fill:{}'.format(p0,p1,color))
        
        return img
        

    
