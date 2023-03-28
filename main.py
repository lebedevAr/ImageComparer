import os.path
import pathlib

import cv2


def get_keypoints(query_img: str, train_img: str):
    # Read the query image as query_img
    # and train image This query image
    # is what you need to find in train image
    # Save it in the same directory
    # with the name image.jpg
    query_img = cv2.imread(query_img)
    train_img = cv2.imread(train_img)

    # Convert it to grayscale
    query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create()

    # Now detect the keypoints and compute
    # the descriptors for the query image
    # and train image
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors, trainDescriptors)

    # draw the matches to the final image
    # containing both the images the drawMatches()
    # function takes both images and keypoints
    # and outputs the matched query image with
    # its train image
    final_img = cv2.drawMatches(query_img, queryKeypoints,
                                train_img, trainKeypoints, matches[:20], None)

    final_img = cv2.resize(final_img, (1000, 650))

    # Show the final image
    cv2.imshow("Matches", final_img)
    cv2.waitKey()


def find_object_on_photo(object_img: str, photo_img: str):
    method = cv2.TM_CCOEFF
    small_image = cv2.imread(object_img)  # что ищу
    large_image = cv2.imread(photo_img)  # где ищу
    result = cv2.matchTemplate(small_image, large_image, method)
    mn, _, mnLoc, _ = cv2.minMaxLoc(result)
    MPx, MPy = mnLoc
    trows, tcols = small_image.shape[:2]
    cv2.rectangle(large_image, (MPx, MPy), (MPx + tcols, MPy + trows), (0, 0, 255), 5)
    res = cv2.resize(large_image, dsize=(2500, 2500))
    cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
    cv2.imshow("Resized", res)
    cv2.waitKey(0)


if __name__ == "__main__":
    first_img = "test_imgs/sber1.jpg"
    second_img = "test_imgs/sber2.png"
    third_img = "test_imgs/sber3.png"
    fourth_img = "test_imgs/sber4.jpg"
    fivth_img = "test_imgs/sber5.jpg"
    sixth_img = "test_imgs/sber6.jpg"
    print(os.path.abspath(sixth_img))
