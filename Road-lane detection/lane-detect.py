import matplotlib.pyplot as plt
import cv2
import numpy as np


# we now mask all the unneccesary points
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)  # define blank matrix that matches image height and weidth
    # we dont need channel conunt because we take in greyscale# channel_count = img.shape[2]  # retrive the no of color channel
    match_mask_color = 255  # create a match color with same color channel
    cv2.fillPoly(mask, vertices, match_mask_color)  # fill inside the polygon which we only want to show
    masked_image = cv2.bitwise_and(img, mask)  # return the image where image pixel matches
    return masked_image


def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) #create black image that match original image

    for line in lines:
        for x1, y1, x2, y2 in line: #line contain four coord-of first point and second point(x1,y1) and then(x2,y2)
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0) #merge two image with weight
    return img


image = cv2.imread('road.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# in matplotlin verticall value increases from top to bottom
print(image.shape)
height = image.shape[0]
width = image.shape[1]
# define region of interest like we want ot find the triangle
# we take bottom left corner and bottom right corner
region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height)
]
# region_of_interest_vertices ::
# (0,height) #bottom left corner
# (width/2,height/2), #middle of the road
# width,height , bottom right corner

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)
cropped_image = region_of_interest(canny_image,
                                   np.array([region_of_interest_vertices], np.int32), )
lines = cv2.HoughLinesP(cropped_image,
                        rho=6,
                        theta=np.pi / 180,
                        threshold=160,
                        lines=np.array([]),
                        minLineLength=40,
                        maxLineGap=25)
image_with_lines = draw_the_lines(image, lines)
plt.imshow(image_with_lines)
# plt.imshow(canny_image)
plt.show()

# plt.imshow(image)
# plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
