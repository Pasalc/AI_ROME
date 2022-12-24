import cv2
from PIL import Image
import numpy as np
from random import randint,uniform
import os

roman_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
           (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

roman_dict = {1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X', 40: 'XL', 50: 'L', 90: 'XC', 100: 'C', 400: 'CD', 500: 'D', 900: 'CM', 1000: 'M'}

def into_roman(num):
    res = ''
    while num > 0:
        for i, r in roman_map:
            while num >= i:
                res += r
                num -= i
    return res

    ad
def get_black_background():
    return np.zeros((500, 500, 3)) 
def generate_image_with_text(text):
    image = get_black_background()
    posX = int(image.shape[0]*uniform(0.1,0.9))
    posY = int(image.shape[1]*uniform(0.1,0.9))
    place=(posX, posY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale=uniform(0.5,2)
    r,g,b=randint(0,255),randint(0,255),randint(0,255)
    if r+g+b==0:
        r=1
    color=(b,g,r)
    thickness=randint(1,4)
    lineType=cv2.LINE_AA
    cv2.putText(image, text,  place, font, fontScale, color, thickness, lineType)
    return image

def generate_dataset(roman_map):
    for i,r in roman_map:
        path="dataset/{}".format(i) #"dataset/"+str(i)+"/"+r+".png"
        os.makedirs(path, exist_ok=True)
        for image_count in range(1000):
            image=generate_image_with_text(r)
            image_path=f"{path}/{image_count}.png"
            print(image_path)
            cv2.imwrite(image_path,image)

if __name__ == "__main__":
    print("Generating dataset")
    generate_dataset(roman_map)
    print("Done")


#num = int(input("Enter a number: "))
#print(into_roman(num))
