#	Importing required libraries
from datetime import datetime
from hashlib import sha256
from requests import get
from urllib.parse import urlparse
from subprocess import Popen, PIPE
from time import time

#	Building a blockchain
class Blockchain:

	def __init__(self):
		'''
		Initializing chain, transactions and nodes.
		
		Arguments:
			self
		Return:
			None
		'''

		self.chain = []
		self.transactions = []	# Contain transactions before going to the block
		self.create_block(proof = 1, prev_hash = '0', timestamp = str(datetime.now()))
		self.nodes = set()	# For faster computation, set is used

	def create_block(self, proof, prev_hash, timestamp):
		'''
		Create a block and append it in the chain.
		
		Arguments:
			self
			proof -> Nonce value -> int
			prev_hash -> Hash of last block in current chain -> str
			timestamp -> Timestamp of the block -> str
		Return:
			The last block of the new chain -> dictionary
		'''
		
		block = {'index': len(self.chain) + 1,
				'timestamp': timestamp,
				'proof': proof,
				'previous_hash': prev_hash,
				'transactions': self.transactions}
		
		self.transactions = []	#	Empty the transactions list as they are already added to the block
		self.chain.append(block)
		
		return block

	def create_dump(self, block):
		'''
		Convert block to string.

		Arguments:
			self
			block -> The block which is to be converted to string -> dictionary
		Return:
			The string form of the block -> str
		'''
		
		string = '{} {} {}'.format(	block['index'],
									block['timestamp'],
									block['previous_hash'])
		for transaction in block['transactions']:
			string += ' {} {} {}'.format(transaction['sender'],
										transaction['reciever'],
										transaction['amount'])
		string += ' {}'.format(block['proof'])
		
		return string

	def create_dump_mine(self, block):
		'''
		Convert block to string for mining.

		Arguments:
			self
			block -> The block which is to be converted to string for mining -> dictionary
		Return:
			The string form of the block -> str
		'''

		string = '{} {} {}'.format(	block['index'],
									block['timestamp'],
									block['previous_hash'])
		for transaction in block['transactions']:
			string += ' {} {} {}'.format(transaction['sender'],
										transaction['reciever'],
										transaction['amount'])
		string += ' '
		
		return string

	def get_previous_block(self):
		'''
		To get the last block of the chain.

		Arguments:
			self
		Return:
			Last block of the chain -> dicttionary
		'''
		
		return self.chain[-1]

	def proof_of_work(self, prev_hash, DEVICE, DIFFICULTY):
		'''
		Calculate the nonce.

		Arguments:
			self
			prev_hash -> Hash of the last block of the current chain -> str
			DEVICE -> Device on which mining is to be done -> str
			DIFFICULTY -> Difficulty of mining -> int
		Return:
			Nonce found -> int
			Timestamp for which nonce is found -> str
		'''

		block = {'index': len(self.chain) + 1,
				'timestamp': str(datetime.now()),
				'proof': 1,
				'previous_hash': prev_hash,
				'transactions': self.transactions}
		
		start_time = time()
		
		if DEVICE == 'GPU':		#	For mining on GPU
			
			nonce = 0
			while nonce == 0:
				block['timestamp'] = str(datetime.now())
				encoded_block = self.create_dump_mine(block)
				args = ('./run', encoded_block, str(DIFFICULTY))
				
				#	Start mining using the executable file
				process = Popen(args, stdout=PIPE)
				if(process.wait() != 0):
					print('Error with executable')
					exit(0)
				nonce = int(process.stdout.read().decode())
			
			block['proof'] = nonce
		
		elif DEVICE == 'CPU':	#	For mining on CPU
			
			check_proof = False
			while not check_proof:	# check_proof is False
				encoded_block = self.create_dump(block).encode()
				new_hash = sha256(encoded_block).hexdigest()
				if new_hash[:DIFFICULTY] == '0'*DIFFICULTY:	# Condition imposed to make mining not lose value
					check_proof = True
					print(new_hash)
				else:
					block['proof'] += 1
					block['timestamp'] = str(datetime.now())
		
		end_time = time()
		print('Execution time: ', (end_time - start_time))
		
		return block['proof'], block['timestamp']

	def hash(self, block):
		'''
		Return hash of the block.

		Arguments:
			self
			block -> The block for which hash is to be calculated -> dictionary
		Return:
			Hash of the block -> str
		'''

		encoded_block = self.create_dump(block).encode()
		return sha256(encoded_block).hexdigest()

	def is_chain_valid(self, DIFFICULTY):
		'''
		Check if the chain is valid or not.
		i.e. Verify that chain has not been tampered with.

		Arguments:
			self
			chain -> The chain which is to be validated -> list of dictionaries
			DIFFICULTY -> The difficulty of mining -> int
		Return:
			True if it is valid, else False -> boolean
		'''

		prev_block = self.chain[0]
		
		for block in self.chain[1:]:
			
			if block['previous_hash'] != self.hash(prev_block):	#	Checking the chain integrity
				return False
			
			new_hash = self.hash(block)
			if new_hash[:int(DIFFICULTY)] != '0'*int(DIFFICULTY):	# Checking the proof of work
				print(new_hash[:int(DIFFICULTY)], '0'*int(DIFFICULTY))
				return False
			
			prev_block = block

		return True

	def add_transaction(self, sender, reciever, amount):
		'''
		Add a transaction to the pool of unverified transactions.

		Arguments:
			self
			sender -> Name of the sender -> str
			reciever -> Name of the reciever -> str
			amount -> The amount transfered -> double
		Return:
			Block index in which transaction is expected to be verified -> int
		'''

		self.transactions.append({'sender': sender,
								'reciever': reciever,
								'amount': amount})

		return self.get_previous_block()['index'] + 1

	def add_node(self, address):
		'''
		Add nodes in the network.

		Arguments:
			self
			address -> Socket to be added in the network -> str
		Return:
			None
		'''
		
		parsed_url = urlparse(address)
		self.nodes.add(parsed_url.netloc)

	def replace_chain(self, DIFFICULTY):
		'''
		Ping each node for its version of chain and decided whether to update own chain or not.

		Arguments:
			self
		Return:
			True, if chain is replaced, else False -> boolean
		'''

		network = self.nodes
		longest_chain = None
		max_length = len(self.chain)
		
		for node in network:
			response = get(f'http://{node}/get_chain')	#	Send request to get chain of the node
			if response.status_code == 200:
				length = response.json()['length']
				chain = response.json()['chain']
				if length > max_length and self.is_chain_valid(DIFFICULTY):
					max_length = length
					longest_chain = chain
		
		if longest_chain:
			self.chain = longest_chain
			
			return True	#	Tellls that chain was replaced
		
		return False