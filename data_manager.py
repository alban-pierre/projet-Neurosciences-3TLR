import pickle

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
				id_1 = [r['id'] for r in self.data]
				id_2 = [r['id'] for r in data]
				for iid in range(len(id_2)):
					if (id_2[iid] not in id_1):
						self.data.append(data[iid])
						
		except:
			print 'Warning : the file "{}" could not be opened to add data.'.format(filename)
	
