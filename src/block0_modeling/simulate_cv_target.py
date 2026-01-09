import numpy as np # numerical computing library (vectors and matrices)

def simulate_target_cv(x0, dt, steps):
    """
    Constant-velocity truth model (for simulation only).
    x = [px, py, vx, py]
    """
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)  # state transition matrix: New position = Old poisition + velocity * dt 
    
    x = x0.copy() # copy of the x0 input vector
    truth = [x.copy()] # python list that stores the sequence of "true states" over time

    for _ in range(steps): 
        x = F @ x # @ means matrix multiplication 
        # F is 4x4, x is 4x1, result is 4x1 expected 
        # we compute x(k+1) = F * x(k)
        truth.append(x.copy())

    return np.array(truth), F  # return truth as a numpy array and the state transition matrix F

def measure_position(truth_states, sigma_pos):
    '''
    truth_states: array of true states over time (shape(N,4))
    sigma_pos: standard deviation of measurement noise (meters)
    ''' 
    noise = np.random.randn(len(truth_states),2) * sigma_pos 
    # randn method creates standard normal distribution mean 0 and standard deviation is 1
    z = truth_states[: ,:2] + noise # this line is array slicing basically
    # columns 0 and 1 are [px, py]
    return z


def main():
    np.random.seed(0) # seeding randomness 
        # this makes random numbers repeatable 
        # Meaning: if you run the script twice, you get the same random noise -> easier debugging 

    dt = 0.1 # time step is 0.1 seconds namely 10 Hz
    steps = 50 # simulate 50 steps -> total time 50 * 0.1 = 5 seconds 

    x0 = np.array([0.0, 0.0, 300.0, 40.0]) # 300 m/s east, 40 m/s north
    # initial target state: start at (0,0) and velocity (300,40) m/s 
    # meaning after 0.1s px increases 30m and py increases by 4m 

    truth, F = simulate_target_cv(x0, dt, steps)
    # truth becomes array of size (51,4)
    # F is 4x4 

    z = measure_position(truth, sigma_pos = 20.0)
    # z becomes an array of size (51,2)
    # each row is noisy measurement of (px,py)

    for k in range(steps+1):
        print("truth[{}] =".format(k), truth[k])
        print("meas[{}] =".format(k), z[k])

if __name__ == "__main__":
    main()

        
    


