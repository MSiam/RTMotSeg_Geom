from metrics import Metrics
import sys
import os
import cv2
import matplotlib.pyplot as plt
import scipy

mm= Metrics(2)

masks_dir= sys.argv[1]
out_dir= sys.argv[2]

for f in sorted(os.listdir(masks_dir)):
    label= scipy.misc.imread(masks_dir+f)
    #import pdb; pdb.set_trace()
    label[label==3]=0
    label[label==1]=0
    label[label==2]=1

    out= scipy.misc.imread(out_dir+f)
    out= scipy.misc.imresize(out, label.shape, 'nearest')
    out[out<128]=0
    out[out>128]=1

    mm.update_metrics(out,label,0,0)
    print('handling file ', f)

mm.compute_rates()
mm.compute_final_metrics(1)
print("Iou "+ str(mm.mean_iou_index))
print('per class iou ', mm.iou)


