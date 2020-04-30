import numpy as np

F32 = 0xFFFFFFFF

_k = np.array([	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
				0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
				0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
				0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
				0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
				0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
				0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
				0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
				0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
				0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
				0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
				0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
				0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
				0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
				0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
				0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2])

def _preprocesspad(s):
	hex_str = ''
	for ch in s:
		temp = hex(ch)[2:]
		if len(temp) == 1:
			temp = '0' + temp
		hex_str += temp
	s = hex_str

	_lenhex = hex(len(s)*4)[2:]
	if len(_lenhex) < 16:
		_lenhex = '0'*(16-len(_lenhex)) + _lenhex
	_blocksize = 8*8*2

	_padlen = 0
	unevenlen = len(s) % _blocksize
	if unevenlen < _blocksize - 16:
		_padlen = _blocksize - unevenlen - 17
	else:
		_padlen = _blocksize*2 - unevenlen - 17

	s = s + '8' + '0'*_padlen + _lenhex

	blocks = np.array([int(s[i:i+8], 16) for i in range(0, len(s), 8)])
	return blocks.reshape(-1, 16)

def _ROTRIGHT(a, b):
	return ((a >> b) | (a << (32-b))) & F32

def _CH(a, b, c):
	return ((a & b) ^ (~a & c)) & F32

def _MAJ(a, b, c):
	return ((a & b) ^ (a & c) ^ (b & c)) & F32

def _EP0(x):
	return (_ROTRIGHT(x, 2) ^ _ROTRIGHT(x, 13) ^ _ROTRIGHT(x, 22)) & F32

def _EP1(x):
	return (_ROTRIGHT(x, 6) ^ _ROTRIGHT(x, 11) ^ _ROTRIGHT(x, 25)) & F32

def _SIG0(x):
	return (_ROTRIGHT(x, 7) ^ _ROTRIGHT(x, 18) ^ (x >> 3)) & F32

def _SIG1(x):
	return (_ROTRIGHT(x, 17) ^ _ROTRIGHT(x, 19) ^ (x >> 10)) & F32

class SHA256PY:
	def __init__(self, m=None):
		self._h = np.array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
							0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19])

		self._w = np.zeros(64, dtype=np.int64)
		self.updateSHA256(m)

	def compressSHA256(self):
		# print('W:', self._w)
		# print('initial: ', hex(self.a), hex(self.b), hex(self.c), hex(self.d), hex(self.e), hex(self.f), hex(self.g), hex(self.h))
		self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h = self._h
		for i in range(0, 64):
			t1 = (self.h + _EP1(self.e) + _CH(self.e, self.f, self.g) + _k[i] + self._w[i]) & F32
			t2 = (_EP0(self.a) + _MAJ(self.a, self.b, self.c)) & F32
			self.h = self.g
			self.g = self.f
			self.f = self.e
			self.e = (self.d + t1) & F32
			self.d = self.c
			self.c = self.b
			self.b = self.a
			self.a = (t1 + t2) & F32

			# print('t='+str(i)+':', hex(self.a), hex(self.b), hex(self.c), hex(self.d), hex(self.e), hex(self.f), hex(self.g), hex(self.h))

		self._h[0] = (self._h[0] + self.a) & F32
		self._h[1] = (self._h[1] + self.b) & F32
		self._h[2] = (self._h[2] + self.c) & F32
		self._h[3] = (self._h[3] + self.d) & F32
		self._h[4] = (self._h[4] + self.e) & F32
		self._h[5] = (self._h[5] + self.f) & F32
		self._h[6] = (self._h[6] + self.g) & F32
		self._h[7] = (self._h[7] + self.h) & F32

	def updateSHA256(self, m):
		if not m.any():
			return

		for j in range(len(m)):
			self._w[:16] = m[j]
			for i in range(16, 64):
				self._w[i] = (_SIG1(self._w[i-2]) + self._w[i-7] + _SIG0(self._w[i-15]) + self._w[i-16]) & F32
			self.compressSHA256()

	def hexdigest(self):
		hash_val = '0x'
		for num in self._h:
			hash_val += hex(num)[2:]
		return hash_val

print(SHA256PY(_preprocesspad('abc'.encode())).hexdigest())