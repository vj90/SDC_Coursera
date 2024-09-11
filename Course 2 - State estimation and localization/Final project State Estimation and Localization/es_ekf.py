# Starter code for the Coursera SDC Course 2 final project.
#
# Author: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt3_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
'''gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
#plt.show()'''

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our 
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
# C_li = np.array([
#      [ 0.9975 , -0.04742,  0.05235],
#      [ 0.04992,  0.99763, -0.04742],
#      [-0.04998,  0.04992,  0.9975 ]
# ])

t_i_li = np.array([0.5, 0.1, 0.5]) # translation vector
# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 1.00 

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian
meas_var = np.eye(6)
for i in range(3):
    meas_var[i,i]=var_imu_f
    meas_var[i+3,i+3]=var_imu_w


#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0
f_jac = np.eye(9)


#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain
    K = p_cov_check@h_jac.T@np.linalg.inv(h_jac@p_cov_check@h_jac.T + sensor_var) 
    #print (K)
    # 3.2 Compute error state
    delta_p = K@(y_k - p_check)
    # 3.3 Correct predicted state
    p_hat = p_check + delta_p[0:3]
    v_hat = v_check + delta_p[3:6]
    delq = Quaternion(euler=delta_p[6:9])
    q_hat = q_check.quat_mult_right(delq,out='Quaternion')
    q_hat=q_hat.normalize()
    #print(q_hat)
    # 3.4 Compute corrected covariance
    p_cov_hat = (np.eye(9)-K@h_jac)@p_cov_check
    return p_hat, v_hat, q_hat, p_cov_hat

#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
# helper function to reshape 1D arrays to vectors
def vec(x):
    return np.reshape(x,(x.shape[0],1))
def getF(f_jac,Cns,acc,delta_t):
    f_jac[0:3,3:6]=delta_t*np.eye(3)
    k = -skew_symmetric(Cns@acc)*delta_t
    f_jac[3:6,6:9]=k
    return f_jac
def getQ(delta_t):
    return delta_t*delta_t*meas_var
def time_approx_equal(t1,t2):
    if abs(t1-t2)<0.001:
        return True
    else:
        return False
    
latest_lidar_idx = 1
latest_GPS_idx = 1

qs=[]
qs=[gt.r[0] for k in range(imu_f.data.shape[0])]
acx = v_est.copy()

for k in range(1, 5770*0+ 1*imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]
    # 1. Update state with IMU inputs
    qkminus1 = Quaternion(q_est[k-1,0],q_est[k-1,1],q_est[k-1,2],q_est[k-1,3])
    Cns = qkminus1.to_mat()
    acc = vec(imu_f.data[k-1])

    pk = vec(p_est[k-1]) + delta_t*vec(v_est[k-1]) + (delta_t**2/2)*(Cns@acc + vec(g)) 
    vk = vec(v_est[k-1]) + delta_t*(Cns@acc + vec(g))
    thetak = imu_w.data[k-1]*delta_t
    #print(180/3.14*thetak)
    delqk = Quaternion(euler=thetak)
    qk = qkminus1.quat_mult_left(delqk,out='Quaternion').normalize()

    if k >115750:
        print('angle before rotation')
        print(qkminus1.to_euler())
        print('angle after rotation')
        print(qkminus1.to_euler()+thetak)
        print('quart angle after rotation')
        print(qk.to_euler())
        print('quart before rotation')
        print(qkminus1)
        print('quart after rotation')
        print(qk,'\n---------------------------')
    # print(Cns@acc + vec(g))
    # print('IMU data before rotation')
    # print(imu_f.data[k-1])
    # print('IMU data after rotation')
    # print(Cns@(np.reshape(imu_f.data[k-1],(3,1))))
    # print('Initial angle')
    # print(gt.r[0])
    # 1.1 Linearize the motion model and compute Jacobians
    f_jac = getF(f_jac,Cns,acc,delta_t)
    Q = getQ(delta_t)
    # 2. Propagate uncertainty
    # print(p_cov[k-1])
    p_cov[k] = f_jac@p_cov[k-1]@f_jac.T + l_jac@Q@l_jac.T
    # print(p_cov[k])
    # 3. Check availability of GNSS and LIDAR measurements
    if latest_lidar_idx<lidar.t.shape[0] and time_approx_equal(imu_f.t[k],lidar.t[latest_lidar_idx]):
        ldata = vec(lidar.data[latest_lidar_idx])
        latest_lidar_idx += 1
        print('Updating lidar')
        pk, vk, qk, p_cov[k] = measurement_update(var_lidar*np.eye(3), p_cov[k], ldata, pk, vk, qk)
    if latest_GPS_idx<gnss.t.shape[0] and  time_approx_equal(imu_f.t[k],gnss.t[latest_GPS_idx]):
        gdata = vec(gnss.data[latest_GPS_idx])
        pk, vk, qk, p_cov[k] = measurement_update(var_gnss*np.eye(3), p_cov[k], gdata, pk, vk, qk)
        latest_GPS_idx += 1
        print('Updating gnss')

    # Update states (save)
    p_est[k] = pk.T
    #print(v_est[0])
    v_est[k] = vk.T
    #print(qk)
    q_est[k] = qk.to_numpy().T
    qs[k]=(qk.to_euler())
    acx[k] = (Cns@acc + vec(g)).T
    # p_cov[k] already set
#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
#plt.show()

################################################################################################
# Plot accelerations and angles
# Create a figure with 6 subplots
fig, axs = plt.subplots(4, 3, figsize=(12, 8),sharex=True)
#time =  [k for k in imu_f.t[0:imu_f.data.shape[0]]]
time =  [k for k in range(imu_f.data.shape[0])]
# Plot data on each subplot
axs[0, 0].plot(time, [k[0] for k in acx], label='aX')
axs[0, 0].plot(time, [k[0] for k in imu_f.data], label='aX')
axs[0, 0].set_title('AX')
axs[0, 1].plot(time, [k[1] for k in acx], label='aY')
axs[0, 1].plot(time, [k[1] for k in imu_f.data], label='aY')
axs[0, 1].set_title('AY')
axs[0, 2].plot(time, [k[2] for k in acx], label='aZ')
axs[0, 2].set_title('AZ')

axs[1, 0].plot(time, [[0] for k in imu_w.data], label='wX')
axs[1, 0].set_title('WX')
axs[1, 1].plot(time, [k[1] for k in imu_w.data], label='wY')
axs[1, 1].set_title('WY')
axs[1, 2].plot(time, [k[2] for k in imu_w.data], label='wZ')
axs[1, 2].set_title('WZ')

axs[2, 0].plot(time, [k[0] for k in v_est], label='vX',marker='.')
axs[2, 0].set_title('VX')
axs[2, 1].plot(time, [k[1] for k in v_est], label='vY',marker='.')
axs[2, 1].set_title('VY')
axs[2, 2].plot(time, [(k[0]**2 + k[1]**2)**0.5 for k in v_est], label='vZ')
axs[2, 2].set_title('VZ')

axs[3, 0].plot(time, [k[0] for k in qs], label='X')
axs[3, 0].set_title('phiX')
axs[3, 1].plot(time, [k[1] for k in qs], label='Y')
axs[3, 1].set_title('phiY')
axs[3, 2].plot(time, [k[2]*180/3.14 for k in qs], label='Z',marker = '.')
axs[3, 2].set_title('phiZ')


# Adjust spacing between subplots
plt.tight_layout() 
#plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################

error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()

#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################

# Pt. 1 submission
# p1_indices = [9000, 9400, 9800, 10200, 10600]
# p1_str = ''
# for val in p1_indices:
#     for i in range(3):
#         p1_str += '%.3f ' % (p_est[val, i])
# with open('pt1_submission.txt', 'w') as file:
#     file.write(p1_str)

# Pt. 2 submission
# p2_indices = [9000, 9400, 9800, 10200, 10600]
# p2_str = ''
# for val in p2_indices:
#     for i in range(3):
#         p2_str += '%.3f ' % (p_est[val, i])
# with open('pt2_submission.txt', 'w') as file:
#     file.write(p2_str)

# Pt. 3 submission
p3_indices = [6800, 7600, 8400, 9200, 10000]
p3_str = ''
for val in p3_indices:
    for i in range(3):
        p3_str += '%.3f ' % (p_est[val, i])
with open('pt3_submission.txt', 'w') as file:
    file.write(p3_str)