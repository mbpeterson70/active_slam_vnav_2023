#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.utils import source
import rasterio
import cv2
import math

def findTransform(input_points,output_points):

    ip = np.float32(input_points)
    op = np.float32(output_points)

    H, mask = cv2.findHomography(ip, op)

    return H

class OrthoImageLoader:
    def __init__(self,pathToTiffFile,forcedCrs=None,forcedExtent=None):
        # forcedExtent: tuple given in order left, right, top, bottom.
        #  - left is the longitude corresponding with x pixel coordinate 0
        #  - right is the longitude corresponding with x pixel coordinate <maximum>
        #  - top is the latitude corresponding with y pixel coordinate 0
        #  - bottom is the longitude corresponding with y pixel coordinate <maximum>

        self.dataset = rasterio.open(pathToTiffFile,'r')

        if (forcedExtent is not None):
            self.left, self.right, self.top, self.bottom = forcedExtent
        else:           
            self.left = self.dataset.bounds.left
            self.right = self.dataset.bounds.right
            self.top = self.dataset.bounds.top
            self.bottom = self.dataset.bounds.bottom

        self.imagedata = self.dataset.read().transpose(1,2,0)

        self.im_h_px, self.im_w_px, self.im_c = self.imagedata.shape

        input_points = np.array([[0,self.im_h_px],[self.im_w_px,self.im_h_px],[0,0],[self.im_w_px,0]])
        output_points = np.array([[self.left,self.bottom],[self.right,self.bottom],[self.left,self.top],[self.right,self.top]])

        self.coordsToPxTransform = findTransform(input_points,output_points)
        self.pxToCoordsTransform = findTransform(output_points,input_points)

        if (forcedCrs is not None):
            self.crs = forcedCrs
        else:
            self.crs = self.dataset.crs

        self.row_mesh = None
        self.col_mesh = None

        self.resolution_m_per_px = (self.right-self.left)/self.im_w_px

    def getMeshgrid(self):
        if (self.row_mesh is None or self.col_mesh is None):
            cols_vect = np.linspace(self.left,self.right,num=self.im_w_px)
            rows_vect = np.linspace(self.top,self.bottom,num=self.im_h_px)
            self.col_mesh,self.row_mesh = np.meshgrid(cols_vect,rows_vect)
        
        return (self.row_mesh, self.col_mesh)

    def getExtent(self):
        return (self.left, self.right, self.bottom, self.top)

    def getImageData(self):
        return self.imagedata
    
    def getImageCoordsAsMapCoords(self, im_r, im_c):
        #vert_coord = (im_r / self.im_h_px) * (self.top-self.bottom) + self.bottom
        #horiz_coord = (im_c / self.im_w_px) * (self.right-self.left) + self.left
        #return (vert_coord,horiz_coord)

        ones = np.ones_like(im_r)
        imageCoords = np.vstack((im_c,im_r,ones))

        mapCoords = np.matmul(self.coordsToPxTransform,imageCoords)
        mapCoords = mapCoords/mapCoords[2,:]

        return mapCoords[0,:], mapCoords[1,:]
    
    def getMapCoordsAsImageCoords(self, x, y):      
        #x_px = (x-self.left)/(self.right-self.left) * self.im_w_px
        #y_px = (y-self.top)/(self.bottom-self.top) * self.im_h_px
        #return(x_px,y_px)

        ones = np.ones_like(x)

        mapCoords = np.vstack((x,y,ones))

        #print("xxxxx mapCoords=")
        #print(mapCoords)

        pxcoords = np.matmul(self.pxToCoordsTransform,mapCoords)
        pxcoords = pxcoords/pxcoords[2,:]

        #print("pxcoords=")
        #print(pxcoords)

        return (np.squeeze(pxcoords[0,:]), np.squeeze(pxcoords[1,:]))

    def plotMap(self,ax,showOnlyRgb=True):
        if (showOnlyRgb):
            imageToShow = self.imagedata[:,:,0:3]
        else:
            imageToShow = self.imagedata

        ax.imshow(imageToShow,extent=[self.left,self.right,self.bottom,self.top],origin='upper')
    
    def getResolution(self):
        (imagedata_h_px, imagedata_w_px, imagedata_d) = self.imagedata.shape
        
        # Compute spatial resolution from width only (assume same resolution in both x and y)
        image_width_mapunits = self.right-self.left
        
        image_resolution_mapunits = image_width_mapunits/imagedata_w_px
        
        return image_resolution_mapunits
    
    def getCrs(self):
        return self.crs


    def getScaledMapImage(self, scaleFactor=0.1):
        # Returns map image scaled to a different resolution. Resolution determined by given factor.

        #self.im_h_px, self.im_w_px

        scaled_h_px = int(scaleFactor * self.im_h_px)
        scaled_w_px = int(scaleFactor * self.im_w_px)

        return cv2.resize(self.imagedata[:,:,0:3], (scaled_w_px, scaled_h_px))


    def getSubimage(self,x,y,theta_deg,scale, output_dim_px, sampleResolution_m_per_px,onlyRgb=False):

        theta = theta_deg/180*np.pi

        output_dim_m = sampleResolution_m_per_px * output_dim_px
        input_dim_px = output_dim_m / self.resolution_m_per_px

        #x_px = (x-self.left)/(self.right-self.left) * self.im_w_px
        #y_px = (self.top-y)/(self.top-self.bottom) * self.im_h_px

        #x_px, y_px = self.getMapCoordsAsImageCoords(x,y)

        #x_px = np.squeeze(x_px)
        #y_px = np.squeeze(y_px)

        # Define the left, right, top, bottom values of unrotated square at origin.
        x_l = -output_dim_m/2*scale
        x_r = output_dim_m/2*scale

        y_b = -output_dim_m/2*scale
        y_t = output_dim_m/2*scale

        # Stack corner coordinates into a matrix
        square_points_map = np.array([[x_l,y_t],[x_r,y_t],[x_r,y_b],[x_l,y_b]]).T

        # Stack pixel coordinates in same order
        #output_px_coord_target = np.array([[0,0],[output_dim_px,0],[output_dim_px,output_dim_px],[0,output_dim_px]]).T
        output_px_coord_target = np.array([[0,output_dim_px],[output_dim_px,output_dim_px],[output_dim_px,0],[0,0]]).T

        # Define rotation matrix
        R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

        # Rotate points and translate by centerpoint
        square_points_map = np.matmul(R,square_points_map) + np.array([[x],[y]])

        square_points_map_x = square_points_map[0,:]
        square_points_map_y = square_points_map[1,:]

        # Find corresponding image coordinates
        square_points_px_x, square_points_px_y = self.getMapCoordsAsImageCoords(square_points_map_x, square_points_map_y)

        # Specify a margin around the minimum subimage
        px_margin = 10

        # For speeding up cropping, take a subimage around pixel coordinates
        subimage_minx = max((0,math.floor(np.amin(square_points_px_x))-px_margin))
        subimage_maxx = min((math.ceil(np.amax(square_points_px_x))+px_margin,self.im_w_px))

        subimage_miny = max((0,math.floor(np.amin(square_points_px_y))-px_margin))
        subimage_maxy = min((math.ceil(np.amax(square_points_px_y))+px_margin,self.im_h_px))

        subimage = self.imagedata[subimage_miny:subimage_maxy,subimage_minx:subimage_maxx,:]

        square_points_px_x_subimage = square_points_px_x - subimage_minx
        square_points_px_y_subimage = square_points_px_y - subimage_miny

        points1 = np.vstack((square_points_px_x_subimage,square_points_px_y_subimage)).T

        h, status = cv2.findHomography(points1, output_px_coord_target.T)

        dsize = (output_dim_px, output_dim_px)

        si = cv2.warpPerspective(subimage,h,dsize)

        # Rotate point coordinates according to desired rotation

        #(subimage, srcPoints, destPoints) = getRotatedSubimage(self.imagedata,x_px,y_px,input_dim_px,output_dim_px,theta,scale,returnSrcAndDestPoints=True)

        if (onlyRgb):
            return si[:,:,0:3]
        else:
            return si