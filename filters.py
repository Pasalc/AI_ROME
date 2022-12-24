import sys
import numpy as np
import pandas #for debug purpose

ALPHA = 0.001

def dRelu(in_volume):
    in_volume*=ALPHA
    return in_volume
def ReLu(in_volume):
	in_volume[ in_volume <0 ] = ALPHA
	return in_volume
class Convolutional:
	#parameter:
		#image_dim = touple of dimension of the image and depth ( H x W x depth)
		#kernels = touple of number of kernels x dimension of the kernel x depth ( n_kern x H x W x depth)
	def __init__(self, image_dim=(1,16,16), kernels=(3,3,1), padding=1, stride=1, bias=0.1, eta=0.01): #eta=learning rate
		if (image_dim[0]!=kernels[3]): 
			print(sys.stderr,"Error: depth of the filter must match the depth of the image.")

		self.specs=kernel 		# Only a tuple with the specifications
		self.padding=padding
		self.stride=stride
		self.eta = eta
		self.iteration = 0 		# To update the gradient descent 
		#self.bias = []   		# List of bias, same for each kernel so one value for each of it (kernels[0])		
		self.cache = 0			# Useful to save a padded image
		#self.filters = np.random.rand(*kernels)*0.1 # The parameters of the neurons aka kernels

		#for i in range(kernels[0]): self.bias.append(bias)

	def pad(image,padding):
		new_img=np.zeros( image.shape )
        new_img[padding:,padding:]=image
        return new_img
    
    def convolve(image,kernel,stride):
        k_x,k_y=kernel.shape
        x_size=image.shape[0]-k_x
        y_size=image.shape[1]-k_y
        out=np.zeros(x_size,y_size)
        for x in range(x_size):
            for y in range(y_size):
                out[x][y]=np.tensordot(image[x:x+k_x][y+s_y:y+k_y],kernel)
	def fwd(self, image):
        """
        if(self.padding!=0): 
			image=pad(self.image,self.padding)
		self.cache=image	#save for backpropagation
        """
		y_out, x_out = 0,0
        out=convolve(image,kernel)
        #out=[]
        #for kernel in self.filters:
        #    out.append(convolve(image,kernel))
		out=np.array(out)
        #out=ReLu(out)
        #self.dW=self.dRelu(out)
        return out #ReLu(out),out
	def bp(self, d_out):
		#INPUT:
			# d_out = d(out_volume) = loss gradient of the output of this conv layer  ( out_W, out_H, out_depth ) 
		#RETURN:
			#d_input 		= loss gradient of the input image received during the convolution (np.array)
			#self.d_kernel	= gradient of the filter (np.array)
        
        flipped_kernel=np.flip(self.kernel)
        padded_d_out=pad(d_out,flipped_kernel.shape[0]-1)
        d_input=convolve(padded_d_out,np.flip(kernel))
        d_kernel=convolve(self.image,np.flip(d_out)) #*self.dW))
        """d_input=[]
        d_filters=[]
        for kernel in self.filters:
            kernel=np.flip(kernel)
            d_input.append(convolve(d_out,kernel))
            d_filters.append(d_out)
        d_input=np.array(d_input)
        d_filter=[]
        """
		return d_input

    def gd(self, d_kernel): #, d_bias):
		self.eta = self.eta * np.exp(-self.iteration/20000)
		self.kernel -= self.eta * (d_kernel)
		#for i in range(len(self.bias)): self.bias[i] -= self.eta * d_bias[i]
		self.iteration +=1

class Pooling: 
	def __init__(self, image_dim=(1, 16,16), mode='avg', size=2, stride=2):
		self.cache=None
		#self.image_dim=image_dim
		self.size=size
		self.stride=stride
		self.mode=mode
	def fwd(self,image): #For Pooling layers, it is not common to pad the input using zero-padding.
		w_in, h_in = self.image_dim.shape

		w_out = int((w_in - self.size)/self.stride)+1
		h_out = int((h_in - self.size)/self.stride)+1

		out=np.zeros((layers, w_out, h_out))

		#for layer in range(layers):
			y_out, x_out = 0,0
			for y in range(0, h_in - self.size, self.stride):
				x_out=0
				for x in range(0, w_in - self.size, self.stride):
					if self.mode=='avg':
						out[ y_out, x_out] = np.average(images[  y:y+self.size, x:x+self.size])
					else: #max pooling is applied
						out[layer, y_out, x_out] = np.max(images[  y:y+self.size, x:x+self.size])
					x_out+=1
				y_out+=1
		return out


	def bp(self, d_out): #d_out is like the derivative of the pooling output, image is the input of the pooling layer
		
		layers, w_in, h_in = self.image_dim

		w_out = int((w_in - self.size)/self.stride)+1
		h_out = int((h_in - self.size)/self.stride)+1

		out = np.zeros((layers, w_in, h_in))

        y_out, x_out = 0,0
        for y in range(0, h_out, self.stride):
            x_out=0

            for x in range(0, w_out, self.stride):

                if self.mode=='avg':	#not sure about that

                    average_dout=d_out[y_out,x_out]/(self.size*2)
                    out[ y:(y+self.size), x:(x+self.size)] += np.ones((self.size,self.size))*average_dout

                else: #max pooling is applied

                    area = self.cache[ y:y+self.size, x:x+self.size]
                    index = np.nanargmax(area)
                    (y_i, x_i) = np.unravel_index(index, area.shape)
                    out[ y + y_i, x + x_i] += d_out[ y_out, x_out]
                x_out+=1
            y_out+=1
		return 	out
