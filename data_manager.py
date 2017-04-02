import pickle
import matplotlib.pyplot as plt
import numpy as np

class Data_manager:

	def __init__(self, filename=''):
		self.filename = filename
		if (len(filename) <= 0):
			self.filename = 'Data_manager_result_file.txt'
		try:
			with open(self.filename, 'r') as file_d:
				depick = pickle.Unpickler(file_d)
				self.data = depick.load()
		except:
			print 'Warning : the file "{}" could not be opened to load the data.'.format(self.filename)
			self.data = []
			
			
	def add_result(self, d):
		self.data.append(d)
		
		
	def change_filename(self, filename):
		if (len(filename) > 0):
			self.filename = filename

			
	def __del__(self):
		try:
			with open(self.filename, 'w') as file_d:
				pick = pickle.Pickler(file_d)
				pick.dump(self.data)
		except:
			print 'Warning : the file "{}" could not be opened to save the data.'.format(filename)
			
			
	def add_results_from_file(self, filename):
		try:
			with open(filename, 'r') as file_d:
				depick = pickle.Unpickler(file_d)
				data = depick.load()
				id_1 = [r[0] for r in self.data]
				id_2 = [r[0] for r in data]
				for iid in range(len(id_2)):
					if (id_2[iid] not in id_1):
						self.data.append(data[iid])
						
		except:
			print 'Warning : the file "{}" could not be opened to add data.'.format(filename)

			
	def plot_results(self, param_x, x, y, plot_points=True, plot_mean=False, plot_std=False, plot_min_max=False, figure=1, x_log_scale=False, pointargs=[], pointkargs=[], meanargs=[], meankargs=[], stdargs=[], stdkargs=[], minmaxargs=[], minmaxkargs=[]):
		x_in_all_results = 0
		param = dict(param_x)
		plt_x = []
		del param[x]
		for r in self.data:
			if r[1].has_key(x):
				if set(param.items()).issubset(r[1].items()):
					plt_x.append(tuple([r[1][x]] + [r[2][ykey] for ykey in y]))
			else:
				x_in_all_results = x_in_all_results + 1
		if (x_in_all_results > 0):
			print 'Warning : the key "{}" was not found in {} results.'.format(x, x_in_all_results)
		plt_x = np.array(sorted(plt_x))
		plt.figure(figure)
		if x_log_scale:
			plt.xscale('log')
		if plot_points:
			for i_plot in range(1,plt_x.shape[1]):
				if (len(pointargs) >= i_plot) and (len(pointkargs) >= i_plot):
					plt.plot(plt_x[:,0], plt_x[:,i_plot], *(pointargs[i_plot-1]), **(pointkargs[i_plot-1]))
				elif (len(pointargs) >= i_plot):
					plt.plot(plt_x[:,0], plt_x[:,i_plot], *(pointargs[i_plot-1]))
				elif (len(pointkargs) >= i_plot):
					plt.plot(plt_x[:,0], plt_x[:,i_plot], **(pointkargs[i_plot-1]))
				else:
					plt.plot(plt_x[:,0], plt_x[:,i_plot])
		if plot_mean:
			plt_ux = sorted(list(set(plt_x[:,0])))
			plt_m = np.zeros((len(plt_ux), plt_x.shape[1]))
			for i in range(len(plt_ux)):
				plt_m[i,:] = plt_x[plt_x[:,0] == plt_ux[i]].mean(axis=0)
			for i_plot in range(1,plt_x.shape[1]):
				if (len(meanargs) >= i_plot) and (len(meankargs) >= i_plot):
					plt.plot(plt_m[:,0], plt_m[:,i_plot], *(meanargs[i_plot-1]), **(meankargs[i_plot-1]))
				elif (len(meanargs) >= i_plot):
					plt.plot(plt_m[:,0], plt_m[:,i_plot], *(meanargs[i_plot-1]))
				elif (len(meankargs) >= i_plot):
					plt.plot(plt_m[:,0], plt_m[:,i_plot], **(meankargs[i_plot-1]))
				else:
					plt.plot(plt_m[:,0], plt_m[:,i_plot])
		plt.show(block=False)
