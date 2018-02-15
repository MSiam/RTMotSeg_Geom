import numpy as np
import sys

def get_weights(nclasses, npy_file):
    yy= np.load(npy_file)
    label_to_frequency= {}
    for c in range(nclasses):
        class_mask= np.equal(yy, c)
        class_mask= class_mask.astype(np.float32)
        label_to_frequency[c]= np.sum(class_mask)

    #perform the weighing function label-wise and append the label's class weights to class_weights
    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        class_weights.append(class_weight)

    return class_weights

def main(path):
    weights= get_weights(2, path+'Y_train.npy')
    np.save(path+'weights.npy',weights)

    xtrain= np.load(path+'X_train.npy')
    np.save(path+'mean.npy',[xtrain[:,:,:,0].mean(),xtrain[:,:,:,1].mean(), xtrain[:,:,:,2].mean()] )

    flotrain= np.load(path+'Flo_train.npy')
    np.save(path+'flo_mean.npy',[flotrain[:,:,:,0].mean(),flotrain[:,:,:,1].mean(), flotrain[:,:,:,2].mean()] )

if __name__=="__main__":
    main(sys.argv[1])
