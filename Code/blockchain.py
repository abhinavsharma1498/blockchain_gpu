# Importing libraries
import datetime
import hashlib
import json
from flask import Flask, jsonify

# Building a blockchain
class Blockchain:

	def __init__(self):
		self.chain = []
		self.create_block(proof = 1, prev_hash = '0', timestamp = str(datetime.datetime.now()))

	def create_block(self, proof, prev_hash, timestamp):
		block = {'index': len(self.chain) + 1,
				'timestamp': timestamp,
				'proof': proof,
				'previous_hash': prev_hash}
		self.chain.append(block)
		return block

	def get_previous_block(self):
		return self.chain[-1]

	def proof_of_work(self, prev_hash, prev_proof):
		block = {'index': len(self.chain) + 1,
				'timestamp': str(datetime.datetime.now()),
				'proof': 1,
				'previous_hash': prev_hash}
		check_proof = False
		while not check_proof:	# check_proof is False
			# Defining an asymmetric problem
			hash_operation = hashlib.sha256(str(block['proof']**2 - prev_proof**2).encode()).hexdigest()
			if new_hash[:4] == '0000':	# Condition imposed to make mining not lose value
				check_proof = True
				print(new_hash)
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
			prev_proof = prev_block['proof']
			proof = block['proof']
			hash_operation = hashlib.sha256(str(proof**2 - prev_proof**2).encode()).hexdigest()
			if hash_operation[:4] != '0000':	# Checking the proof of work
				return False
			prev_block = block
			block_index += 1
		return True

# Creating a Web App
app = Flask(__name__)

# Creating a Blockchain
blockchain = Blockchain()

# Mining a a new block
@app.route('/mine_block', methods = ['GET'])	# Eg: http://127.0.0.1:5050/mine_block
def mine_block():
	prev_block = blockchain.get_previous_block()
	prev_proof = prev_block['proof']
	prev_hash = blockchain.hash(prev_block)
	proof, timestamp = blockchain.proof_of_work(prev_hash, prev_proof)
	block = blockchain.create_block(proof = proof, prev_hash = prev_hash, timestamp = timestamp)
	response = {'message': 'Your block is mined!!',
				'index': block['index'],
				'timestamp': block['timestamp'],
				'proof': block['proof'],
				'previous_hash': block['previous_hash']}
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

# Running the app
app.run(host = '0.0.0.0', port = 5000)	# Public host and port 5000