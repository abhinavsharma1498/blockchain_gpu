#	Importing libraries
from flask import Flask, jsonify, request
from uuid import uuid4
from argparse import ArgumentParser
from blockchain import Blockchain

#	Defining the constants
DIFFICULTY = 2
PORT = 5000
DEVICE = 'CPU'

#	Parsing the command line arguments
parse = ArgumentParser(description='Run blockchain server for this machine')
parse.add_argument('--port', '-p', metavar='PORT', type=int, help='The port number for the server', required=True)
parse.add_argument('--difficulty', '-d', type=int, help='The difficulty of hashing for Mining')
parse.add_argument('--device', '-D', type=str, help='Delect device: CPU or GPU', default='CPU')
cmd_args = parse.parse_args()

if cmd_args.difficulty:
	DIFFICULTY = cmd_args.difficulty

PORT = cmd_args.port
if PORT < 1024:
	print('Invalid port number')
	exit(0)

if cmd_args.device:
	DEVICE = cmd_args.device.upper()
if not (DEVICE == 'CPU' or DEVICE == 'GPU'):
	print('Enter a valid device name')
	exit(0)

#	Creating a Web App
app = Flask(__name__)

#	Creating an address for the node and port
node_address = str(uuid4()).replace('-', '')

#	Creating a Blockchain
blockchain = Blockchain()

#	Mining a new block
@app.route('/mine_block', methods = ['GET'])	#	Eg: http://127.0.0.1:5050/mine_block
def mine_block():
	'''
	Mine a new block for the chain.

	Arguments:
		None
	Return:
		Response message -> json oblect
		HTTP SUCCESS OK code -> int
	'''

	prev_block = blockchain.get_previous_block()
	prev_proof = prev_block['proof']
	prev_hash = blockchain.hash(prev_block)
	
	blockchain.add_transaction(sender = node_address, reciever = 'Abhinav', amount = 10)	#	Taking the fees of transaction
	proof, timestamp = blockchain.proof_of_work(prev_hash, DEVICE, DIFFICULTY)
	block = blockchain.create_block(proof = proof, prev_hash = prev_hash, timestamp = timestamp)
	
	response = {'message': 'Your block is mined!!',
				'index': block['index'],
				'timestamp': block['timestamp'],
				'proof': block['proof'],
				'previous_hash': block['previous_hash'],
				'transactions': block['transactions']}
	
	return jsonify(response), 200

#	Getting full blockchain
@app.route('/get_chain', methods = ['GET'])		#	Eg: http://127.0.0.1:5050/get_chain
def get_chain():
	'''
	Display the full chain.

	Arguments:
		None
	Return:
		Response message containing the chain and its length -> json object
		HTTP SUCCESS OK code -> int
	'''

	response = {'chain': blockchain.chain,
				'length': len(blockchain.chain)}
	
	return jsonify(response), 200

#	Checking if chain is valid
@app.route('/is_valid', methods = ['GET'])
def is_valid():
	'''
	Check if the chain is valid or not.

	Arguments:
		None
	Return:
		Response message -> json object
		HTTP SUCCESS OK code -> int
	'''
	
	if blockchain.is_chain_valid(DIFFICULTY):
		response = {'message': 'Blockchain is valid.'}
	else:
		response = {'message': 'Blockchain is not valid.'}
	
	return jsonify(response), 200

#	Adding a new transaction to blockchain
@app.route('/add_transaction', methods = ['POST'])
def add_transaction():
	'''
	Adding transaction to the pool.

	Arguments:
		None
	Return:
		Response message -> json object
		HTTP SUCCESS OK/ERROR code -> int
	'''

	json = request.get_json()
	transaction_keys = ['sender', 'reciever', 'amount']
	
	if not all (key in json for key in transaction_keys):
		response = {'message': 'Some elements of the transaction are missing.'}
		
		return jsonify(response), 400
	
	index = blockchain.add_transaction(json['sender'], json['reciever'], json['amount'])
	response = {'message': f'Your transaction will be added to block {index}',}
	
	return jsonify(response), 201

#	Decentralize blockchain

#	Connecting new node
@app.route('/connect_node', methods = ['POST'])
def connect_node():
	'''
	Connect new nodes to the network.

	Arguments:
		None
	Return:
		Response message -> json object
		HTTP SUCCESS OK/ERROR code -> int
	'''

	json = request.get_json()
	nodes = json.get('nodes')
	if nodes is None:
		response = {'message': 'No nodes were found'}
		
		return jsonify(response), 400
	
	for node in nodes:
		blockchain.add_node(node)
	
	response = {'message': 'All nodes are now connnected. the blockchain now contains following nodes:',
				'total_nodes': list(blockchain.nodes)}
	
	return jsonify(response), 201

#	Replace chain by longest chain, if needed
@app.route('/replace_chain', methods = ['GET'])
def replace_chain():
	'''
	Applying Consensus Protocol for updating the chain.

	Arguments:
		None
	Return:
		Response message -> json object
		HTTP SUCCESS OK/ERROR code -> int
	'''
	
	if blockchain.replace_chain():
		response = {'message': 'The node had different chains, so the chain was replaced by largest one.',
					'new_chain': blockchain.chain}
	else:
		response = {'message': 'All good.',
					'actual_chain': blockchain.chain}
	
	return jsonify(response), 200

#	Running the app
app.run(host='0.0.0.0', port=PORT)	#	Public host
