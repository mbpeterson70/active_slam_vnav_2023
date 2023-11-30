#!/usr/bin/python3
#!/usr/bin/env python3

import gtsam

class SegmentSLAM():
    
    def __init__(self, K, distortion_params):
        self.cal3ds2 = gtsam.Cal3DS2(K[0,0], K[1,1], K[0,1],self.u0,self.v0,self.k1,self.k2,self.p1,self.p2)
    