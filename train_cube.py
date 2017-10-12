import pycuber as pc
from random import randint
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape, Permute, Embedding, Add, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import VarianceScaling
from keras import backend as K
from keras.models import load_model
from path import Path
import keras.backend as K
import threading
import itertools
import time

np.random.seed(1337)
max_moves =  10
cache_dir = Path('./cache')

mycube = pc.Cube()
faces = ['L','U','R','D','F','B']
colors = ['[r]','[y]','[o]','[w]','[g]','[b]']
possible_moves = ["R","R'","R2","U","U'","U2","F","F'","F2","D","D'","D2","B","B'","B2","L","L'","L2"]




def sol2cat(solution):
    # transform solution to one hot vector encoding
    # first map move to number, then genereate one hot encoding
    # using keras utils
    
    global possible_moves
    sol_tmp = []
    for j in range(len(solution)):
        sol_tmp.append(possible_moves.index(solution[j]))
        
    sol_cat = to_categorical(sol_tmp)
    
    return sol_cat
    
    

def cube2np(mycube):
    # transform cube object to np array
    # works around the weird data type used
    global faces
    global colors
    cube_np = np.zeros((6,3,3))
    for i,face in enumerate(faces):
        face_tmp = mycube.get_face(face)
        for j in range(3):
            for k in range(len(face_tmp[j])):
                caca = face_tmp[j][k]
                cube_np[i,j,k] = colors.index(str(caca))
    return cube_np

def generate_game(max_moves = max_moves):
    
    # generate a single game with max number of permutations number_moves
    
    mycube = pc.Cube()

    global possible_moves
    formula = []
    cube_original = cube2np(mycube)
    number_moves = max_moves#randint(3,max_moves)

    action = randint(0,len(possible_moves)-1)
    for j in range(number_moves):
        formula.append(possible_moves[action])
        new_action = randint(0,len(possible_moves)-4)
        delta = action % 3
        action_face = action - delta
        if new_action >= action_face:
            new_action += 3
        action = new_action
        
    #my_formula = pc.Formula("R U R' U' D' R' F R2 U' D D R' U' R U R' D' F'")

    my_formula = pc.Formula(formula)


    mycube = mycube((my_formula))
    # use this instead if you want it in OG data type

    cube_scrambled = mycube
    
    solution = my_formula.reverse()

    #print(mycube)


    return cube_scrambled,solution

def generate_N_games(N=10,max_moves=max_moves):
    
    
    scrambled_cubes = []
    solutions = []
    for j in range(N):
        cube_scrambled,solution = generate_game(max_moves = max_moves)
        scrambled_cubes.append(cube_scrambled)
        solutions.append(solution)
        
    return scrambled_cubes,solutions

def generate_action_space(number_games=100):
    D = [] # action space
    game_count = 0
    play_game = True
    global max_moves
    while play_game:


        scrambled_cube,solutions = generate_game(max_moves = max_moves)

        state = scrambled_cube
        for j in range(len(solutions)):
            action = solutions[j]
            current_state = state.copy()
            state_next = state(action)
            D.append((current_state,action))
            state = state_next

        game_count+=1

        if game_count>=number_games:
            break
            
    return D

class GenerateExample(threading.Thread):
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, N=1):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        threading.Thread.__init__(self)
        self.N = N

    def run(self):
        """ Method that runs forever """
        while True:
            N = self.N
            x  = []
            y = []

            D = generate_action_space(N)
            for d in D:
                x.append(cube2np(d[0]))
                y.append(to_categorical(possible_moves.index((str(d[1]))),len(possible_moves)))

            x = np.asarray(x)
            x = x.reshape(x.shape[0],54,)
            x = x.astype('float32')

            y = np.asarray(y)
            y = y.reshape(y.shape[0],y.shape[2])

            num_files = len(cache_dir.files()) // 2

            np.save(cache_dir / str(num_files) + '_x.npy', x)
            np.save(cache_dir / str(num_files) + '_y.npy', y)
    
striper = lambda t: t[:-5]
vstrip = np.vectorize(striper)

def agreg_files():
    x_array = np.array(cache_dir.files('*_x.npy'))
    y_array = np.array(cache_dir.files('*_y.npy'))

    x_array.sort()
    y_array.sort()

    if len(x_array) != len(y_array):
        time.sleep(.1)
        return agreg_files()
    else:
        if np.prod(vstrip(x_array) == vstrip(y_array)) != 1:
            time.sleep(.1)
            return agreg_files()

    try:
        x_array = np.concatenate([np.load(file) for file in x_array])
        y_array = np.concatenate([np.load(file) for file in y_array])
    except Exception as e:
        time.sleep(.1)
        return agreg_files()

    return x_array, y_array

def gate(x):
    splt = K.tf.split(x, 2, axis=-1)
    return splt[0] * K.sigmoid(splt[1])

batch_size = 1024
num_classes = len(possible_moves)
num_epochs = 100
hidden_units = int(162/.698)
hidden_units = 1024 // 4
num_classes_embedding = 3
input_shape = (54,)
augmented = False
use_gate = False
use_res = False
use_teacher = False
teacher_strengh = 1/2

model = Sequential()
model.add(Embedding(6, num_classes_embedding, input_length=54))
model.add(Reshape((num_classes_embedding*54,)))

if not use_gate:
    if not use_res:
        model.add(Dense(hidden_units,
                        kernel_initializer='he_normal',
                        activation='relu'))
    else:
        model.add(Lambda(lambda x: Add()([x/np.sqrt(2), Dense(hidden_units,kernel_initializer='he_normal')(Activation('relu')(x))/np.sqrt(2)])))
else:
    if not use_res:
        model.add(Dense(int(hidden_units*.7)*2,
                        kernel_initializer=VarianceScaling(2),
                        input_shape=input_shape))
        model.add(Lambda(gate))
    else:
        model.add(Lambda(lambda x: Add()([x/np.sqrt(2), gate(Dense(hidden_units,kernel_initializer=VarianceScaling(2))(x))/np.sqrt(2)])))
for i in range(0):
    if not use_gate:
        if not use_res:
            model.add(Dense(hidden_units,
                            kernel_initializer='he_normal',
                            activation='relu'))
        else:
            model.add(Lambda(lambda x: Add()([x/np.sqrt(2), Dense(hidden_units,kernel_initializer='he_normal')(Activation('relu')(x))/np.sqrt(2)])))
    else:
        if not use_res:
            model.add(Dense(int(hidden_units*.7)*2,
                            kernel_initializer=VarianceScaling(2),
                            input_shape=input_shape))
            model.add(Lambda(gate))
        else:
            model.add(Lambda(lambda x: Add()([x/np.sqrt(2), gate(Dense(hidden_units,kernel_initializer=VarianceScaling(2))(x))/np.sqrt(2)])))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

if not use_teacher:
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
else:
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

if use_teacher:
    teacher_model = load_model('rubiks_model_bak3.h5')
# model.load_weights('rubiks_model_bak3.h5')

x, y = agreg_files()
for file in cache_dir.files():
    file.remove()
np.save(cache_dir / str(0) + '_x.npy', x)
np.save(cache_dir / str(0) + '_y.npy', y)
if use_teacher:
    y = teacher_model.predict(x, verbose=1, batch_size=batch_size*8)*teacher_strengh + to_categorical(y, num_classes)*(1-teacher_strengh)
else:
    y = np.expand_dims(y, -1)
for j in range(num_epochs):
    model.fit(x, y, epochs=j+1,verbose=1,initial_epoch =j,
              batch_size=batch_size)#generate_data(8)
    num_training = len(x)
    new_x, new_y = agreg_files()
    for file in cache_dir.files():
        file.remove()
    np.save(cache_dir / str(0) + '_x.npy', new_x)
    np.save(cache_dir / str(0) + '_y.npy', new_y)
    if use_teacher:
        new_x = new_x[num_training:]
        new_y = new_y[num_training:]
        new_y = teacher_model.predict(new_x, verbose=1, batch_size=batch_size*8)*teacher_strengh + to_categorical(new_y, num_classes)*(1-teacher_strengh)
        x = np.concatenate([x, new_x])
        y = np.concatenate([y, new_y])
    else:
        x = new_x
        y = new_y
        y = np.expand_dims(y, -1)
    print(model.evaluate(x[num_training:], y[num_training:], batch_size=batch_size*8))
    model.save('rubiks_model.h5')  # creates a HDF5 file 'my_model.h5'




