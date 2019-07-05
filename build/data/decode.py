import numpy as np
import base64 as b64
import PIL.Image as Image

if __name__ == "__main__":
    secret = CODE_HERE
    tips = b64.b64decode(secret.encode()).decode()
    list1 = []
    list1 = tips[1:len(tips)-1].split(",")
    list2 = []
    for i in list1:
        list2.append(int(i))
    print(len(list2))
    m = np.array(list2)
    n = m.size
    narray = m.reshape([128, 128])
    array = np.asarray(narray, dtype=np.uint8)
    image = Image.fromarray(array).convert('L')
    image.show()
