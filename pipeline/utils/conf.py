from json_minify import json_minify
import json

class Conf:
    
	def __init__(self, confPath):
		# carrega e aloca a configuração e atualiza o dict do objeto
		conf = json.loads(json_minify(open(confPath).read()))
		self.__dict__.update(conf)

	def __getitem__(self, k):
		# return o valor associado à chave fornecida
		return self.__dict__.get(k, None)
