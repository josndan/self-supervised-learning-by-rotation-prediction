import numpy as np

def IOU(image,box):
  x,y,w,h = box
  x,y,w,h = round(x),round(y),round(w),round(h)
  
  union1 = h * w

  intersection = np.count_nonzero(image[x:x+w,y:y+h] !=0)

  i1 = image[:x,:y]  !=0
  i2 = image[x:x+w,:y]  !=0
  i3 = image[x+w:,:y]  !=0
  i4 = image[:x,y:y+h]  !=0
  i5 = image[x+w:,y:y+h]  !=0
  i6 = image[:x,y+h:]   !=0
  i7 = image[x:x+w,y+h:]  !=0
  i8 = image[x+w:,y+h:]  !=0

  union2 = np.count_nonzero(i1) + np.count_nonzero(i2)+ np.count_nonzero(i3)
  + np.count_nonzero(i4)+ np.count_nonzero(i5)+ np.count_nonzero(i6)
  + np.count_nonzero(i7)+ np.count_nonzero(i8)

  return intersection/(union1+union2)