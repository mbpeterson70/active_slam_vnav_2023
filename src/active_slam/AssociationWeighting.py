import matplotlib.pyplot as plt
import numpy as np

def weightAssociation(arithmeticMean, maxDiff):
    
    weight = (1+np.cos(arithmeticMean/maxDiff*np.pi))*(arithmeticMean < maxDiff)*.5
    
    return weight

if __name__=="__main__":
    x = np.linspace(0,1,100)

    maxDiff = 0.5

    weight = weightAssociation(x, maxDiff)

    fig, ax = plt.subplots()
    ax.plot(x,weight)
    plt.show()