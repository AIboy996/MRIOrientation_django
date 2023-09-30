import numpy as np

def ori(image: np.ndarray, target: str) -> np.ndarray:
    match target:
        case "000":
            return image  # 000 Target[x,y,z]=Source[x,y,z]
        case "001":
            return np.fliplr(image)  # 001 Target[x,y,z]=Source[sx-x,y,z]
        case "010":
            return np.flipud(image)  # 010 Target[x,y,z]=Source[x,sy-y,z]
        case "011":
            return np.flipud(np.fliplr(image))  # 011 Target[x,y,z]=Source[sx-x,sy-y,z]
        case "100":
            return image.transpose()  # 100 Target[x,y,z]=Source[y,x,z]
        case "101":
            # 101 Target[x,y,z]=Source[sx-y,x,z] 110 Target[x,y,z]=Source[y,sy-x,z]
            return np.flipud(image.transpose())
        case "110":
            # 110 Target[x,y,z]=Source[y,sy-x,z] 101 Target[x,y,z]=Source[sx-y,x,z]
            return np.fliplr(image.transpose())
        case "111":
            # 111 Target[x,y,z]=Source[sx-y,sy-x,z]
            return np.flipud(np.fliplr(image.transpose()))  
