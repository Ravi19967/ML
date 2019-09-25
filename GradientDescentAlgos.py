import numpy as np
import matplotlib.pyplot as plt

X= [0.5,2.5]
Y= [0.2,0.9]
epochs = 1000
delta = 0.025
batch_size = 2

nestrov_gd = []
momentum_loss = []
w_ngd = []
b_ngd = []
w_gd = []
b_gd = []
m_w = []
m_b = []
gd_loss = []
w_sgd = []
b_sgd = []
sgd_loss = []
w_mb = []
b_mb = []
mb_loss = []
w_adagrad = []
b_adagrad = []
adagrad_loss = []
w_rmsprop = []
b_rmsprop = []
rmsprop_loss = []
w_adadelta = []
b_adadelta = []
adadelta_loss = []
w_adam = []
b_adam = []
adam_loss = []
w_adamax = []
b_adamax = []
adamax_loss = []
w_nadam = []
b_nadam = []
nadam_loss = []

def sigmoid_func(w,b,x):
	return (1)/(1+np.exp(-1*(w*x + b)))

def loss_func(w,b,A,B):
	loss_res = 0
	for x,y in zip(A,B):
		pred = sigmoid_func(w,b,x)
		loss_res += 0.5*(pred-y)**2
	return loss_res

def loss_func_stochastic(w,b,A,B):
	loss_res = 0
	pred = sigmoid_func(w,b,A)
	loss_res += 0.5*(pred-B)**2
	return loss_res

def grad_stochastic(w,b,x_a,y_b):
	grad_delta=np.zeros((2, 1))
	pred = sigmoid_func(w,b,x_a)
	grad_delta[0][0] += (pred)*(1-pred)*(pred - y_b)*x_a
	grad_delta[1][0] += grad_delta[0][0]/x_a
	return grad_delta

def grad_array(w,b,A,B):
	grad_delta=np.zeros((2, 1))
	for x,y in zip(A,B):
		pred = sigmoid_func(w,b,x)
		grad_delta[0][0] += (pred)*(1-pred)*(pred - y)*x
		grad_delta[1][0] += grad_delta[0][0]/x
	return grad_delta

def gd_algo():
	w=-1
	b=-1
	learning_rate = 0.1
	for i in range(epochs):
		grad_res = np.zeros((2,1))
		grad_res = grad_array(w,b,X,Y)
		w -= learning_rate*grad_res[0][0]
		b -= learning_rate*grad_res[1][0]
		w_gd.append(w)
		b_gd.append(b)
		gd_loss.append(loss_func(w,b,X,Y))
		
def Momentum_gd():
	w=-1
	b=-1
	learning_rate = 0.1
	momentum = 0.9
	update = np.zeros((2,1))
	for i in range(epochs):
		grad_res = np.zeros((2,1))
		grad_res = grad_array(w,b,X,Y)
		update = momentum*(update) + learning_rate*grad_res
		w -= update[0][0]
		b -= update[1][0]
		m_w.append(w)
		m_b.append(b)
		momentum_loss.append(loss_func(w,b,X,Y))

def Nesterov_Acc_gd():
	w=-1
	b=-1
	learning_rate = 0.1
	momentum = 0.9
	update = np.zeros((2,1))
	for i in range(epochs):
		grad_res = np.zeros((2,1))
		update = momentum*(update)
		grad_res = grad_array(w - update[0][0],b - update[1][0],X,Y)
		update = momentum*update + learning_rate*grad_res
		w -= update[0][0]
		b -= update[1][0]
		w_ngd.append(w)
		b_ngd.append(b)
		nestrov_gd.append(loss_func(w,b,X,Y))

def SGD():
	w=-1
	b=-1
	learning_rate = 0.1
	for i in range(epochs):
		grad_res = np.zeros((2,1))
		for a_x,b_y in zip(X,Y):
			grad_res = grad_stochastic(w,b,a_x,b_y)
			w -= learning_rate*grad_res[0][0]
			b -= learning_rate*grad_res[1][0]
			w_sgd.append(w)
			b_sgd.append(b)
			sgd_loss.append(loss_func_stochastic(w,b,a_x,b_y))

def MiniBatch():
	w=-1
	b=-1
	learning_rate = 0.1
	for i in range(epochs):
		batch_size_curr = 0
		grad_res = np.zeros((2,1))
		batch_x = []
		batch_y = []
		for a_x,b_y in zip(X,Y):
			grad_res += grad_stochastic(w,b,a_x,b_y)
			batch_x.append(a_x)
			batch_y.append(b_y)
			batch_size_curr += 1
			if(batch_size_curr == 2):
				grad_res = grad_array(w,b,batch_x,batch_y)
				w -= learning_rate*grad_res[0][0]
				b -= learning_rate*grad_res[1][0]
				grad_res = np.zeros((2,1))
				batch_size_curr = 0
				w_mb.append(w)
				b_mb.append(b)
				mb_loss.append(loss_func(w,b,batch_x,batch_y))
				batch_x = []
				batch_y = []

def Adagrad():
	w=-1
	b=-1
	learning_rate = 0.1
	epislon = 0.01
	grad_sum = np.zeros((2,1))
	for i in range(epochs):
		grad_res = np.zeros((2,1))
		grad_res = grad_array(w,b,X,Y)
		grad_sum += grad_res**2
		w -= (learning_rate/(grad_sum[0][0] + epislon)**0.5)*grad_res[0][0]
		b -= (learning_rate/(grad_sum[0][0] + epislon)**0.5)*grad_res[1][0]
		w_adagrad.append(w)
		b_adagrad.append(b)
		adagrad_loss.append(loss_func(w,b,X,Y))

def RMSProp():
	w=-1
	b=-1
	learning_rate = 0.1
	epislon = 0.01
	update_sum = np.zeros((2,1))
	grad_sum = np.zeros((2,1))
	mom = 0.9
	for i in range(epochs):
		grad_res = np.zeros((2,1))
		grad_res = grad_array(w,b,X,Y)
		grad_sum = mom*(grad_sum**2) + (1-mom)*(grad_res**2)
		w -= (learning_rate/(grad_sum[0][0])**0.5)*grad_res[0][0]
		b -= (learning_rate/(grad_sum[1][0])**0.5)*grad_res[1][0]
		w_rmsprop.append(w)
		b_rmsprop.append(b)
		rmsprop_loss.append(loss_func(w,b,X,Y))

def AdaDelta():
	w=-1
	b=-1
	update_sum = np.zeros((2,1))
	grad_sum = np.zeros((2,1))
	mom = 0.9
	for i in range(epochs):
		grad_res = np.zeros((2,1))
		grad_res = grad_array(w,b,X,Y)
		grad_sum = mom*(grad_sum**2) + (1-mom)*(grad_res**2)
		w -= (update_sum[0][0]/((grad_sum[0][0])**0.5))*grad_res[0][0]
		b -= (update_sum[1][0]/((grad_sum[1][0])**0.5))*grad_res[1][0]
		grad_res[0][0] = (update_sum[0][0]/(grad_sum[0][0]**0.5))*grad_res[0][0]
		grad_res[1][0] = (update_sum[1][0]/(grad_sum[1][0]**0.5))*grad_res[1][0]
		update_sum = mom*(update_sum**2) + (1-mom)*(grad_res**2)
		w_adadelta.append(w)
		b_adadelta.append(b)
		adadelta_loss.append(loss_func(w,b,X,Y))

def Adam():
	w=-1
	b=-1
	learning_rate = 0.1
	epislon = 0.1
	mean_moment = np.zeros((2,1))
	mean_moment_beta1 = 0.1
	variance_moment = np.zeros((2,1))
	variance_moment_beta2 = 0.1
	for i in range(epochs):
		grad_res = np.zeros((2,1))
		grad_res = grad_array(w,b,X,Y)
		mean_moment = mean_moment_beta1*mean_moment + (1 - mean_moment_beta1)*(grad_res)
		variance_moment = variance_moment_beta2*variance_moment + (1 - variance_moment_beta2)*((grad_res)**2)
		variance_res = variance_moment/(1 - variance_moment_beta2**(i+1))
		mean_res = mean_moment/(1 - mean_moment_beta1**(i+1))
		w -= (learning_rate/(variance_res[0][0])**0.5 + epislon)*mean_res[0][0]
		b -= (learning_rate/(variance_res[1][0])**0.5 + epislon)*mean_res[1][0]
		w_adam.append(w)
		b_adam.append(b)
		adam_loss.append(loss_func(w,b,X,Y))

def AdaMax():
	w=-1
	b=-1
	learning_rate = 0.1
	epislon = 0.1
	mean_moment = np.zeros((2,1))
	mean_moment_beta1 = 0.1
	variance_moment = np.zeros((2,1))
	variance_moment_beta2 = 0.1
	for i in range(epochs):
		grad_res = np.zeros((2,1))
		grad_res = grad_array(w,b,X,Y)
		mean_moment = mean_moment_beta1*mean_moment + (1 - mean_moment_beta1)*(grad_res)
		variance_moment = np.maximum(variance_moment_beta2*variance_moment,abs(grad_res))
		mean_res = mean_moment/(1 - mean_moment_beta1)
		w -= (learning_rate/(variance_moment[0][0]))*mean_res[0][0]
		b -= (learning_rate/(variance_moment[1][0]))*mean_res[1][0]
		w_adamax.append(w)
		b_adamax.append(b)
		adamax_loss.append(loss_func(w,b,X,Y))

def Nadam():
	print("here")

def Contour():
	x = np.arange(-5.0, 5.0, delta)
	y = np.arange(-5.0, 5.0, delta)
	C, D = np.meshgrid(x, y)
	Z=(loss_func(C,D,X,Y))

	for i in range(epochs):
		plt.legend(["Momentum Based Loss","Gradient Loss","Nestrov Accelerated Loss","Stochastic Loss","Mini Batch","Adagrad Loss","Adadelta Loss","RMSProp", "Adam Loss","Adamax Loss","Nadam"],loc= "upper left")
		CS = plt.contour(C, D, Z,levels =30,linestyles = "dotted")
		plt.clabel(CS, inline=1, fontsize=10)
		wi = m_w[i]
		bi = m_b[i]
		y = momentum_loss[i]
		plt.scatter(wi,bi, y,color='red')
		wi = w_gd[i]
		bi = b_gd[i]
		y = gd_loss[i]
		plt.scatter(wi,bi,y,color='blue')
		wi = w_ngd[i]
		bi = b_ngd[i]
		y = nestrov_gd[i]
		plt.scatter(wi,bi,y,color='green')
		wi = w_sgd[i]
		bi = b_sgd[i]
		y = sgd_loss[i]
		plt.scatter(wi,bi,y,color='pink')
		wi = w_mb[i]
		bi = b_mb[i]
		y = mb_loss[i]
		plt.scatter(wi,bi,y,color='black')
		wi = w_adagrad[i]
		bi = b_adagrad[i]
		y = adagrad_loss[i]
		plt.scatter(wi,bi,y,color ='yellow')
		wi = w_adadelta[i]
		bi = b_adadelta[i]
		y = adadelta_loss[i]
		plt.scatter(wi,bi,y,color ='grey')
		wi = w_rmsprop[i]
		bi = b_rmsprop[i]
		y = rmsprop_loss[i]
		plt.scatter(wi,bi,y)
		wi = w_adam[i]
		bi = b_adam[i]
		y = adam_loss[i]
		plt.scatter(wi,bi,y,color ='magenta')
		wi = w_adamax[i]
		bi = b_adamax[i]
		y = adamax_loss[i]
		plt.scatter(wi,bi,y,color ='brown')
		#wi = w_nadam[i]
		#bi = b_nadam[i]
		#y = nadam_loss[i]
		#plt.scatter(wi,bi,y)
		plt.pause(0.05)

def start():
	Nadam()
	AdaMax()
	Adam()
	RMSProp()
	AdaDelta()
	Adagrad()
	MiniBatch()
	SGD()
	Nesterov_Acc_gd()
	Momentum_gd()
	gd_algo()
	Contour()

start()