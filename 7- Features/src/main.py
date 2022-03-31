import cv2
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity  as compare_ssim


def image_reconstruction(attack1_color, halftone_color, attack2_color):
    # Convert to grayscale.
    img1 = cv2.cvtColor(attack1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(halftone_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    # matches = matches[:int(len(matches) * 100)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(attack2_color,
                                          homography, (width, height))

    # Save the output.
    # cv2.imwrite(f'images/outputs/{img_name}.bmp', transformed_img)

    return transformed_img, len(matches)




if __name__ == "__main__":
    for i in range(1, 5):

        img_name = "{}".format(i)
        original_index = i

        attack1 = cv2.imread(f"images/Attack 1/{img_name}.bmp")  # Image to be aligned.
        halftone = cv2.imread("images/Reference.bmp")  # Reference image.
        attack2 = cv2.imread(f"images/Attack 2/{img_name}.bmp")
        original = cv2.imread("images/Original.bmp")

        transformed_image, match_count = image_reconstruction(attack1, halftone, attack2)
        cv2.imwrite(f'images/outputs/7_1_1/{i}.bmp', transformed_image)

        transformed_gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        ssim_score, ssim_dif = compare_ssim(original_gray, transformed_gray, full=True)
        mse_score = compare_mse(original_gray, transformed_gray)

        with open("result.txt", 'a') as file:
            file.write("\n------------------\n{}.bmp\nSSIM = {}\nMSE = {}\nMatch Point = {}\n".format(i,ssim_score, mse_score, match_count))


