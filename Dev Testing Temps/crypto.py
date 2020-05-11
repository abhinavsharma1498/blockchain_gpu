# Importing libraries
import datetime
import hashlib
import json
from flask import Flask, jsonify, request
import requests
from uuid import uuid4
from urllib.parse import urlparse

# Building a blockchain
class Blockchain:

	def __init__(self):
		self.chain = []
		self.transactions = []	# Contain transactions before going to the block
		self.create_block(proof = 1, prev_hash = '0', timestamp = str(datetime.datetime.now()))
		self.nodes = set()	# For faster computation, set is used

	def create_block(self, proof, prev_hash, timestamp):
		block = {'index': len(self.chain) + 1,
				'timestamp': timestamp,
				'proof': proof,
				'previous_hash': prev_hash,
				'transactions': self.transactions}
		self.transactions = []	# Empty the transactions list as they are already added to the block
		self.chain.append(block)
		return block

	def get_previous_block(self):
		return self.chain[-1]

	def proof_of_work(self, prev_hash, prev_proof):
		block = {'index': len(self.chain) + 1,
				'timestamp': str(datetime.datetime.now()),
				'proof': 1,
				'previous_hash': prev_hash,
				'transactions': self.transactions}
		check_proof = False
		while not check_proof:	# check_proof is False
			# Defining an asymmetric problem
			hash_operation = hashlib.sha256(str(block['proof']**2 - prev_proof**2).encode()).hexdigest()
			new_hash = self.hash(block)
			if new_hash[:4] == '0000':	# Condition imposed to make mining not lose value
				check_proof = True
			else:
				block['proof'] += 1
		return block['proof'], block['timestamp']

	def hash(self, block):
		# Convert block to string using json, to create a json dump
		encoded_block = json.dumps(block, sort_keys = True).encode()
		return hashlib.sha256(encoded_block).hexdigest()

	def is_chain_valid(self, chain):
		prev_block = chain[0]
		block_index = 1
		while block_index < len(chain):
			block = chain[block_index]
			if block['previous_hash'] != self.hash(prev_block):	# Checking the chain integrity
				return False
			new_hash = self.hash(block)
			if new_hash[:4] != '0000':	# Checking the proof of work
				return False
			prev_block = block
			block_index += 1
		return True

	def add_transaction(self, sender, reciever, amount):
		self.transactions.append({'sender': sender,
								'reciever': reciever,
								'amount': amount})
		prev_block = self.get_previous_block()
		return prev_block['index'] + 1	# Index of block that will recieve these transactions

	def add_node(self, address):
		parsed_url = urlparse(address)
		self.nodes.add(parsed_url.netloc)

	def replace_chain(self):
		network = self.nodes
		longest_chain = None
		max_length = len(self.chain)
		for node in network:
			response = requests.get(f'http://{node}/get_chain')	# Send request to get chain of the node
			if response.status_code == 200:
				length = response.json()['length']
				chain = response.json()['chain']
				if length > max_length and self.is_chain_valid(chain):
					max_length = length
					longest_chain = chain
		if longest_chain:
			self.chain = longest_chain
			return True	# Tellls that chain was replaced
		return False

# Creating a Web App
app = Flask(__name__)

# Creating an address for the node and port
node_address = str(uuid4()).replace('-', '')

# Creating a Blockchain
blockchain = Blockchain()

# Mining a a new block
@app.route('/mine_block', methods = ['GET'])	# Eg: http://127.0.0.1:5050/mine_block
def mine_block():
	prev_block = blockchain.get_previous_block()
	prev_proof = prev_block['proof']
	prev_hash = blockchain.hash(prev_block)
	blockchain.add_transaction(sender = node_address, reciever = 'Abhinav', amount = 10)	# Taking the fees of transaction
	proof, timestamp = blockchain.proof_of_work(prev_hash, prev_proof)
	block = blockchain.create_block(proof = proof, prev_hash = prev_hash, timestamp = timestamp)
	response = {'message': 'Your block is mined!!',
				'index': block['index'],
				'timestamp': block['timestamp'],
				'proof': block['proof'],
				'previous_hash': block['previous_hash'],
				'transactions': block['transactions']}
	return jsonify(response), 200	# Response in JSON format, HTTP SUCCESS OK code

# Getting full blockchain
@app.route('/get_chain', methods = ['GET'])	# Eg: http://127.0.0.1:5050/get_chain
def get_chain():
	response = {'chain': blockchain.chain,
				'length': len(blockchain.chain)}
	return jsonify(response), 200

# Checking if chain is valid
@app.route('/is_valid', methods = ['GET'])
def is_valid():
	is_valid = blockchain.is_chain_valid(blockchain.chain)
	if is_valid:
		response = {'message': 'Blockchain is valid.'}
	else:
		response = {'message': 'Blockchain is not valid.'}
	return jsonify(response), 200

# Adding a new transaction to blockchain
@app.route('/add_transaction', methods = ['POST'])
def add_transaction():
	json = request.get_json()
	transaction_keys = ['sender', 'reciever', 'amount']
	if not all (key in json for key in transaction_keys):
		return {'Some elements of the transaction are missing.'}, 400	# Code 400: Error
	index = blockchain.add_transaction(json['sender'], json['reciever'], json['amount'])
	response = {'message': f'Your transaction will be added to block {index}',}
	return jsonify(response), 201	# Code 201: Something was created

# Decentralize blockchain

# Connecting new node
@app.route('/connect_node', methods = ['POST'])
def connect_node():
	json = request.get_json()
	nodes = json.get('node')
	if nodes is None:
		return 'No node found', 400
	for node in nodes:
		blockchain.add_node(node)
	response = {'message': 'All nodes are now connnected. the blockchain now contains following nodes:',
				'total_nodes': list(blockchain.nodes)}
	return jsonify(response), 201

# Replace chain by longest chain, if needed
@app.route('/replace_chain', methods = ['GET'])
def replace_chain():
	is_chain_replaced = blockchain.replace_chain()
	if is_chain_replaced:
		response = {'message': 'The node had different chains, so the chain was replaced by largest one.',
					'new_chain': blockchain.chain}
	else:
		response = {'message': 'All good.',
					'actual_chain': blockchain.chain}
	return jsonify(response), 200

# Running the app
app.run(host = '0.0.0.0', port = 5000)	# Public host and port 5000