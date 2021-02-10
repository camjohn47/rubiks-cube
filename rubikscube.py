import numpy as np
import itertools
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly
import random
import sys

class RubiksCube:
	def __init__(self, length=3, distance=0, seed=None):
		'''
		Arguments: 
		size (int): Determines the length of all cube faces. Ie, each face
		will have size^2 many squares.

		distance (int): The number of random moves used to generate the cube's 
		starting position. 

		seed (int): Optional argument that can be used to fix the random moves
		used to generate the position. When distance and seed are fixed, the same
		starting position will be generated. 
		'''

		self.face_colors = {'front': 'yellow', 'right': 'red', 'back': 'green',
						    'left':'orange', 'top':'blue', 'bottom':'white'}
		self.faces = list(self.face_colors.keys())
		self.face_area = length ** 2
		num_faces = 6
		self.num_squares = num_faces * self.face_area
		self.face_indices = {self.faces[i]: i for i in range(num_faces)}
		self.color_labels = {color: color[0] for color in self.face_colors.values()}
		self.label_colors = {label: color for color, label in self.color_labels.items()}
		self.grid = list(itertools.product(np.arange(length), np.arange(length)))
		self.faces = np.array([[[self.color_labels[self.face_colors[face]]
								 for i in range(length)] for j in range(length)]
								 for face in self.face_colors])

		self.front, self.right, self.back, self.left, self.top, self.bottom = self.faces[0:6, :, :]
		self.invert = lambda index: abs(2-index)
		self.face_radius = 0.5
		directions, dims, layers = [True, False], range(length), range(length)
		self.moves = list(itertools.product(dims, layers, directions))

		# The None move entails doing nothing. It's included so that the search 
		# algorithm isn't limited to finding solutions of the given search depth. 
		self.moves.insert(0, None)
		self.compressions = []
		self.face_to_axis = {'front': 1, 'back': 1, 'left': 0, 'right': 0,
							 'top': 2, 'bottom': 2}
		self.reward_hist = {}
		self.prev_moves = []
		self.tried_moves = defaultdict(set)
		self.move_history = []

		self.build_face_to_cartestian()
		self.x_dom = np.linspace(-3, 6, 10)
		self.y_dom = np.linspace(-3, 6, 10)
		self.z_dom = np.linspace(-3, 6, 10)
		self.input_faces = {'xy': {True: [3, 0, 1, 2], False:[1, 2, 3, 0]},
							'xz': {True: [5, 0, 4, 2], False: [4, 2, 5, 0]}}
		self.output_faces = {'xy': range(4), 'xz': [0, 4, 2, 5]}
		self.input_cols = {'xz': {True: lambda col: [col, col, col, self.invert(col)],
								  False: lambda col: [self.invert(col), self.invert(col), col, col]}}
		self.output_cols = {'xz': {True: lambda col: [col, col, self.invert(col), col],
								   False: lambda col: [col, self.invert(col), self.invert(col), col]}}
		self.flip_faces = {'xz': {True: lambda col: [col, self.invert(col)],
								  False: lambda col: [self.invert(col), col]}}
		self.flip_cols = {'xz': {True: lambda col: [col, self.invert(col)],
								 False: lambda col: [self.invert(col), col]}}
		self.rotations = {tuple([0, 0, True]): ['top', -1], tuple([0, 0, False]): ['top', 1],
				    tuple([0, 2, True]): ['bottom', -1], tuple([0, 2, False]): ['bottom', 1],
					tuple([1, 0, True]): ['left', 1], tuple([1, 0, False]): ['left', -1],
					tuple([1, 2, True]): ['right', -1], tuple([1, 2, False]): ['right', 1],
					tuple([2, 0, True]): ['front', 1], tuple([2, 0, False]): ['front', -1],
					tuple([2, 2, True]): ['back', -1], tuple([2, 2, False]): ['back', 1]}

		self.random_position(distance, seed)

	def build_face_to_cartestian(self):
		'''
		Builds a mapping from the facial row-column coordinates used by the search
		algorithms to the xyz coordinates used by animation: face, row, col -> xyz.
		Also builds the centroid vertex coordinates, which the animation uses to 
		discretize the cube into faces and squares.
		'''

		def vertices_from_center(center, r, plane):
			'''
			Arguments: 
			center (float list or np array): xyz coordinates of a square's center.

			r (float): Square's radius = distance from its center to any vertex.

			plane (str): Name of the 2D plane onto which the square will be projected.
						 Needs to be in ['xy', 'xz', 'yz'] for proper functioning.
			'''

			shifts = ([[r, 0, -r], [r, 0, r], [-r, 0, r], [-r, 0, -r]] if plane == 'xz'
			 		  else [[r, -r, 0], [r, r, 0], [-r, r, 0], [-r, -r, 0]] if plane == 'xy'
			 		  else [[0, r, -r], [0, r, r], [0, -r, r], [0, -r, -r]])
			vertices = np.array([center + shift for shift in shifts])

			return vertices
					
		self.face_to_cartesian = {}
		self.centroid_vertices = {}
		self.face_transf = {'front': ['+', [3], '-', 'xz'], 'back': ['-', [0], '-', 'xz'],
						    'right': [[3], '-', '-', 'yz'], 'left': [[0], '+', '-', 'yz'],
						    'top': ['+', '+', [3], 'xy'], 'bottom': ['+', '-', [0], 'xy']}
		self.dom_mapping = {'-': np.linspace(2.5, 0.5, 3), '+': np.linspace(0.5, 2.5, 3)}

		for face, transf in self.face_transf.items():
			face_index = self.face_indices[face]
			x_transf, y_transf, z_transf, plane = transf
			x_dom = self.dom_mapping[x_transf] if type(x_transf) != list else x_transf
			y_dom = self.dom_mapping[y_transf] if type(y_transf) != list else y_transf
			z_dom = self.dom_mapping[z_transf] if type(z_transf) != list else z_transf
			xyz_dom = np.array(list(itertools.product(x_dom, y_dom, z_dom)))

			for row, col in self.grid:
				dom_index = row + 3*col 
				face_coords = tuple([face_index, row, col])
				xyz_coords = xyz_dom[dom_index]
				self.face_to_cartesian[face_coords] = xyz_coords
				self.centroid_vertices[tuple(xyz_coords)] = vertices_from_center(xyz_coords, self.face_radius, plane)

	def animate(self, title=None):
		'''
		Creates a 3D plot of the cube in plotly. 
		'''

		fig = go.Figure(data=[
		    go.Mesh3d(
		        x=self.x_dom,
		        y=self.y_dom,
		        z=self.z_dom,
		        name=str(title)
		    	)
			])

		for face, i in self.face_indices.items():
			for row, col in self.grid:
				face_coords = tuple([i, row, col])
				cart_coords = self.face_to_cartesian[face_coords]
				color = self.label_colors[self.faces[i, row, col]]
				vertices = self.centroid_vertices[tuple(cart_coords)]
				x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
				fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
									       surfacecolor = color,
									       surfaceaxis = self.face_to_axis[face], 
										   mode = "markers+text", 
										   marker = dict(
            								size = 10,
            								color = color,
            								symbol = 'square',
            								line = dict(width=10, color = 'black')
        								   )))

		fig.update_layout(
			showlegend=False,
			title = {'text': "Rubik's Cube Solver"},
			scene = dict(xaxis = dict(title= 'Width', showticklabels=True),
	        yaxis = dict(title = 'Depth', showticklabels=True),
	        zaxis = dict(title = 'Height', showticklabels=True))
		    )
		fig.update_scenes(xaxis_autorange='reversed')
		fig.show()

	def build_step(self, index, num_moves):
		step = dict(method = 'update', 
					args = [{"visible": [False] * self.total_squares * num_moves},
					              dict(mode= 'immediate',
                                   frame = dict(duration=40, redraw= True),
                                   transition = dict(duration=25))]
                              ,
					label = f"Move {index}" if index != 0 else "Start")
		for i in range(index * self.total_squares, (index + 1) * self.total_squares):
			step['args'][0]['visible'][i] = True

		return step

	def animate_moves(self, moves):
		'''
		Creates a 3D slideshow of the cube as each move from moves is taken. 
		Used by the solve method to produce a slideshow of the cube's solution.
		'''

		fig = go.Figure(data=[
		    go.Mesh3d(
		        x=self.x_dom,
		        y=self.y_dom,
		        z=self.z_dom
		    	)
			])
		self.total_squares = 54
		num_moves = len(moves)
		steps = []
		for move_ind, move in enumerate(moves):
			self.make_move(move)
			for face, i in self.face_indices.items():
				for row, col in self.grid:
					face_coords = tuple([i, row, col])
					cart_coords = self.face_to_cartesian[face_coords]
					color = self.label_colors[self.faces[i, row, col]]
					vertices = self.centroid_vertices[tuple(cart_coords)]
					x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
					fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
											  surfacecolor = color, 
											  surfaceaxis = self.face_to_axis[face], 
											  mode = "markers+text", 
						 					  marker=dict(
	            								size=0,
	            								color = color,
	            								symbol = 'square',
	            								line=dict(width=10, color = 'black')
	        								)))
			step = self.build_step(move_ind, num_moves)
			steps.append(step)
		
		sliders = [dict(active=17, 
                				transition= dict(duration= 0 ), steps=steps)]
		fig.layout.update(sliders=sliders)
		fig.show()
		fig.update_layout(
			showlegend=False,
			title = {'text': "Rubik's Cube Solver"},
			scene = dict(xaxis = dict(title= 'width', showticklabels=True, range = [-3, 6]),
	        yaxis = dict(title = 'depth', showticklabels=True, range = [-3, 6]),
	        zaxis = dict(title = 'height', showticklabels=True, range = [-3, 6]))
		    )
		fig.show()

	def rotate_face(self, face, direction):
		face_index = self.face_indices[face]
		self.faces[face_index][:][:] = np.rot90(self.faces[face_index][:][:],
											    k = direction)

	def xy_transf(self, row, clockwise):
		input_faces = self.input_faces['xy'][clockwise]
		output_faces = self.output_faces['xy']
		self.faces[output_faces, row, :] = np.copy(self.faces[input_faces, row, :]) 

	def xz_transf(self, col, clockwise):
		input_faces = self.input_faces['xz'][clockwise]
		input_cols = self.input_cols['xz'][clockwise](col)
		output_faces = self.output_faces['xz']
		output_cols = self.output_cols['xz'][clockwise](col)
		self.faces[output_faces, :, output_cols] = np.copy(self.faces[input_faces, :, input_cols])

		flip_faces = self.flip_faces['xz'][clockwise](col)
		flip_cols = self.flip_faces['xz'][clockwise](col)
		self.faces[flip_faces, :, flip_cols] = np.flip(self.faces[flip_faces, :, flip_cols])

	def yz_transf(self, aisle, clockwise):
		inv_aisle = self.invert(aisle)

		if clockwise:
			self.top[inv_aisle, :], self.right[:, aisle], self.bottom[aisle, :], self.left[:, inv_aisle] = np.copy(self.right[:, aisle]), np.flip(np.copy(self.bottom[aisle, :])), np.copy(self.left[:, inv_aisle]), np.flip(np.copy(self.top[inv_aisle, :]))
		else:	
			self.top[inv_aisle, :], self.right[:, aisle], self.bottom[aisle, :], self.left[:, inv_aisle] = np.flip(np.copy(self.left[:, inv_aisle])), np.copy(self.top[inv_aisle, :]), np.flip(np.copy(self.right[:, aisle])), np.copy(self.bottom[aisle, :])
	
	def compress(self):
		compression = ''.join([x for x in self.faces.flatten()])
		return compression

	def load_compression(self, compression):
		compression_faces = np.array([[compression[i] for i in range(face*self.face_area, (face+1) * self.face_area)] 
						  for face in range(6)]).reshape(6,3,3)
		self.faces = compression_faces
		self.front, self.right, self.back, self.left, self.top, self.bottom = self.faces[0:6, :, :]

	def make_move(self, move):
		compression = self.compress()
		self.compressions.append(compression)
		self.move_history.append(move)
		self.tried_moves[compression].add(move)

		if not move:
			return

		dim, index, clockwise = move
		move_transfs = [self.xy_transf, self.xz_transf, self.yz_transf]
		move_transf = move_transfs[dim]
		move_transf(index, clockwise)

		if move in self.rotations:
			rot_face, rot_dir = self.rotations[move]
			self.rotate_face(rot_face, rot_dir)

	def make_moves(self, moves):
		for move in moves:
			self.make_move(move)

	def undo_move(self):
		compression = self.compressions.pop(-1)
		self.load_compression(compression)
		move = self.move_history.pop(-1)
		self.tried_moves[compression].remove(move)

	def is_solved(self):
		solved = self.reward() == 0

		return solved

	def get_entropy(self, colors):
		'''
		Returns the entropy of colors, which is the list of colors found on 
		one of the cube's faces. Higher entropy means that the face is farther away
		from being solved, as this indicates the face's colors are more randomly
		distributed and unordered. Lowest entropy is zero, which in this context
		means the face has only one color. This is why reward is defined as the 
		negative of this output (entropy).
		'''

		color_counts = Counter(colors)
		valid_colors = [color for color in color_counts if color.isalpha()]
		counts = [color_counts[color] for color in valid_colors]
		total_counts = float(sum(counts))
		probs = [float(count)/total_counts for count in counts]
		entropy = -1.0* sum([prob * np.log(prob) for prob in probs])

		return entropy

	def reward(self):
		'''
		Returns the average negative entropy of the color distributions found on 
		the cube's faces. Has a maximum at 0, which occurs if and only if all faces
		have exactly one color -> the cube is solved.  
		'''

		compression = self.compress()

		if compression in self.reward_hist:
			return self.reward_hist[compression]

		face_entropies = [self.get_entropy(np.array2string(self.faces[face, :, :])) 
						  for face in self.face_indices.values()]
		avg_entropy = np.average(face_entropies)
		reward = -avg_entropy
		self.reward_hist[compression] = reward

		return reward

	def maximize_reward(self, depth=0, max_depth=3):
		'''
		Depth-first-search for finding the sequence of moves that maximizes
		the cube's reward, which in this case, is equivalent to minimizing its 
		future entropy. If there are multiple sequences with the same minimal reward, 
		the shortest one is returned. 

		Arguments 
		depth (int): The current depth of the search.

		max_depth (int): The maximum depth the search can reach. If this depth is
						 reached, the reward of the cube's current state is returned.

		Returns
		The maximum reward found across all available moves sequences of size 
		max_depth - depth from the cube's current state. This max reward is returned 
		with the moves that must be taken to reach this reward, followed by the depth 
		at which the reward was found.
		'''

		reward = self.reward()

		if depth == max_depth or reward == 0:

			return reward, [], 0

		max_reward, max_moves, min_depth = -np.float('inf'), [], np.float('inf')
		compression = self.compress()
		new_moves = [move for move in self.moves
					 if move not in self.tried_moves[compression]]

		final_reward = -np.float('inf')
		zero_reward_depth = None

		for move in new_moves:
			self.make_move(move)
			reward, reward_moves, reward_depth = self.maximize_reward(depth + 1, max_depth=max_depth)
			reward_moves.insert(0, move)
			reward_depth = len([move for move in reward_moves if move])
			self.undo_move()

			if max_reward < reward:
				max_reward, min_depth = reward, reward_depth
				max_moves = reward_moves

			# Ensures that minimum sized path to completion is followed. 
			elif max_reward == reward and reward_depth < min_depth:
				max_reward, min_depth = reward, reward_depth
				max_moves = reward_moves

			if max_reward == 0 and depth > 0:
				return max_reward, max_moves, reward_depth

		return max_reward, max_moves, reward_depth

	def solve(self, depth=4):
		'''
		Solves the Rubik's Cube by repeatedly maximizing reward until the cube
		is solved. 
		'''

		solution = []

		while not self.is_solved():
			reward, moves, depth = self.maximize_reward(max_depth=depth)
			solution += moves

			for move in moves:
				self.make_move(move)

		# Temporarily undo moves so that they can be retaken by the animation 
		# algorithm.
		for move in solution:
			self.undo_move()

		solution = [move for move in solution if move]
		self.animate_moves(solution)

	def random_position(self, dist, seed=None):
		for i in range(dist):
			move = self.random_move(seed + i)
			self.make_move(move)

	def random_move(self, seed):
		if seed:
			random.seed(seed)

		move = random.sample(self.moves, 1)[0]

		return move



