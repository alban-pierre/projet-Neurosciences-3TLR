import pickle
import matplotlib.pyplot as plt
import numpy as np

class Data_manager:

	def __init__(self, filename=''):
		# We load previously computed results
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
		# Adds a result to that object
		self.data.append(d)
		
		
	def change_filename(self, filename):
		# Change the filename where we are going to store results at the end
		if (len(filename) > 0):
			self.filename = filename

			
	def __del__(self):
		# Saves the object on the disk
		try:
			with open(self.filename, 'w') as file_d:
				pick = pickle.Pickler(file_d)
				pick.dump(self.data)
		except:
			print 'Warning : the file "{}" could not be opened to save the data.'.format(filename)
			
			
	def add_results_from_file(self, filename):
		# Add all results from a file to the current object
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

			
	def plot_results(self, param_x, x, y, plot_points=True, plot_mean=False, plot_std=False, plot_minmax=False, figure=1, x_log_scale=False, pointargs=[], pointkargs=[], meanargs=[], meankargs=[], stdargs=[], stdkargs=[], minmaxargs=[], minmaxkargs=[]):
		# Plots y as a function of x, and takes every points already stored in this object
		# We plot points that contains param_x in their parameters, except for x where we take all x possible 
		x_in_all_results = 0
		param = dict(param_x)
		plt_x = []
		del param[x]
		# Computes previously stored results that include param_x in their parameters
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
		# Plot the points, with plot arguments pointargs and pointkargs
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
		# Plot the mean of points, with plot arguments meanargs and meankargs
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
		# Plot the standard deviation of points, with plot arguments stdargs and stdkargs
		if plot_std:
			plt_ux = sorted(list(set(plt_x[:,0])))
			plt_p = np.zeros((len(plt_ux), plt_x.shape[1]))
			plt_m = np.zeros((len(plt_ux), plt_x.shape[1]))
			for i in range(len(plt_ux)):
				plt_p[i,:] = plt_x[plt_x[:,0] == plt_ux[i]].mean(axis=0) + plt_x[plt_x[:,0] == plt_ux[i]].std(axis=0)
				plt_m[i,:] = plt_x[plt_x[:,0] == plt_ux[i]].mean(axis=0) - plt_x[plt_x[:,0] == plt_ux[i]].std(axis=0)
			for i_plot in range(1,plt_x.shape[1]):
				if (len(stdargs) >= i_plot) and (len(stdkargs) >= i_plot):
					plt.plot(plt_p[:,0], plt_p[:,i_plot], *(stdargs[i_plot-1]), **(stdkargs[i_plot-1]))
					plt.plot(plt_m[:,0], plt_m[:,i_plot], *(stdargs[i_plot-1]), **(stdkargs[i_plot-1]))
				elif (len(stdargs) >= i_plot):
					plt.plot(plt_p[:,0], plt_p[:,i_plot], *(stdargs[i_plot-1]))
					plt.plot(plt_m[:,0], plt_m[:,i_plot], *(stdargs[i_plot-1]))
				elif (len(stdkargs) >= i_plot):
					plt.plot(plt_p[:,0], plt_p[:,i_plot], **(stdkargs[i_plot-1]))
					plt.plot(plt_m[:,0], plt_m[:,i_plot], **(stdkargs[i_plot-1]))
				else:
					plt.plot(plt_p[:,0], plt_p[:,i_plot])
					plt.plot(plt_m[:,0], plt_m[:,i_plot])
		# Plot the minimum and maximum of points, with plot arguments minmaxargs and minmaxkargs
		if plot_minmax:
			plt_ux = sorted(list(set(plt_x[:,0])))
			plt_min = np.zeros((len(plt_ux), plt_x.shape[1]))
			plt_max = np.zeros((len(plt_ux), plt_x.shape[1]))
			for i in range(len(plt_ux)):
				plt_min[i,:] = plt_x[plt_x[:,0] == plt_ux[i]].min(axis=0)
				plt_max[i,:] = plt_x[plt_x[:,0] == plt_ux[i]].max(axis=0)
			for i_plot in range(1,plt_x.shape[1]):
				if (len(minmaxargs) >= i_plot) and (len(minmaxkargs) >= i_plot):
					plt.plot(plt_min[:,0], plt_min[:,i_plot], *(minmaxargs[i_plot-1]), **(minmaxkargs[i_plot-1]))
					plt.plot(plt_max[:,0], plt_max[:,i_plot], *(minmaxargs[i_plot-1]), **(minmaxkargs[i_plot-1]))
				elif (len(minmaxargs) >= i_plot):
					plt.plot(plt_min[:,0], plt_min[:,i_plot], *(minmaxargs[i_plot-1]))
					plt.plot(plt_max[:,0], plt_max[:,i_plot], *(minmaxargs[i_plot-1]))
				elif (len(minmaxkargs) >= i_plot):
					plt.plot(plt_min[:,0], plt_min[:,i_plot], **(minmaxkargs[i_plot-1]))
					plt.plot(plt_max[:,0], plt_max[:,i_plot], **(minmaxkargs[i_plot-1]))
				else:
					plt.plot(plt_min[:,0], plt_min[:,i_plot])
					plt.plot(plt_max[:,0], plt_max[:,i_plot])
		plt.xlabel(x)
		plt.ylabel(' and '.join(y))
		plt.show(block=False)
