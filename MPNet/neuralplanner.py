import argparse
from copy import copy

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import load_test_dataset
from model import MLP
from torch.autograd import Variable
import math
import time
import pinocchio as pino
import matplotlib.pyplot as plt

size = 5.0
N = 14
model_name = 'mlp_500paths_bs64_stride1_PReLU'

# Load trained model for path generation
mlp = MLP(2 * N, N)  # simple @D
mlp.load_state_dict(torch.load(f'models/paper/{model_name}_final.pkl'))

if torch.cuda.is_available():
    mlp.cuda()

path = "../datasets/paper/train/"
# load test dataset
paths, path_lengths = load_test_dataset(path)

urdf_path = "./iiwa_striker.urdf"
pino_model = pino.buildModelFromUrdf(urdf_path)
pino_data = pino_model.createData()

Q_DOT_LIMITS = np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562], dtype=np.float32)


def IsInCollision(x, idx):
    x = x[:7]
    pino.forwardKinematics(pino_model, pino_data, np.concatenate([x, np.zeros(2)], axis=-1))
    xyz = pino_data.oMi[-1].translation
    if np.abs(xyz[-1] - 0.1505) > 0.02:
        return True
    return False


def plot_path(path):
    DISCRETIZATION_STEP = 0.01
    resultant_path = []
    for i in range(len(path) - 1):
        start = path[i].numpy()
        end = path[i + 1].numpy()
        dist = np.sqrt(np.sum(np.square(end - start)))
        if dist > 0:
            increment_total = dist / DISCRETIZATION_STEP
            num_segments = int(math.floor(increment_total))

            q0 = start[:7]
            qk = end[:7]
            dq0 = start[7:]
            dqk = end[7:]
            a_0 = q0
            a_1 = 3 * q0 + dq0
            a_3 = qk
            a_2 = 3 * qk - dqk

            for i in range(num_segments):
                t = i / num_segments
                q = a_3 * t ** 3 + a_2 * t ** 2 * (1 - t) + a_1 * t * (1 - t) ** 2 + a_0 * (1 - t) ** 3
                resultant_path.append(q)
    resultant_path = np.array(resultant_path)
    plt.subplot(131)
    for i in range(6):
        plt.plot(resultant_path[:, i], label=f"q_{i}")
    plt.legend()
    plt.subplot(132)
    xyzs = []
    for i in range(len(resultant_path)):
        x = resultant_path[i, :7]
        pino.forwardKinematics(pino_model, pino_data, np.concatenate([x, np.zeros(2)], axis=-1))
        xyz = copy(pino_data.oMi[-1].translation)
        xyzs.append(xyz)
    xyzs = np.array(xyzs)
    plt.plot(xyzs[:, -1])
    plt.subplot(133)
    plt.plot(xyzs[:, 0], xyzs[:, 1])
    plt.show()


def steerToPoly(start, end, idx):
    DISCRETIZATION_STEP = 0.01
    start = start.numpy()
    end = end.numpy()
    dist = np.sqrt(np.sum(np.square(end - start)))
    if dist > 0:
        increment_total = dist / DISCRETIZATION_STEP
        num_segments = int(math.floor(increment_total))

        q0 = start[:7]
        qk = end[:7]
        dq0 = start[7:]
        dqk = end[7:]
        a_0 = q0
        a_1 = 3 * q0 + dq0
        a_3 = qk
        a_2 = 3 * qk - dqk

        for i in range(num_segments):
            t = i / num_segments
            q = a_3 * t ** 3 + a_2 * t ** 2 * (1 - t) + a_1 * t * (1 - t) ** 2 + a_0 * (1 - t) ** 3
            # q_dot = 3 * a_3 * t ** 2 + a_2 * (-3 * t ** 2 + 2 * t) + a_1 * (
            #        3 * t ** 2 - 4 * t + 1) + a_0 * 3 * (1 - t) ** 2
            # if np.any(q_dot[:6] > Q_DOT_LIMITS):
            #    return 0

            if IsInCollision(q, idx):
                return 0

        if IsInCollision(end, idx):
            return 0
    return 1


def steerTo(start, end, idx):
    DISCRETIZATION_STEP = 0.01
    dists = np.zeros(N, dtype=np.float32)
    for i in range(0, N):
        dists[i] = end[i] - start[i]

    distTotal = 0.0
    for i in range(0, N):
        distTotal = distTotal + dists[i] * dists[i]

    distTotal = math.sqrt(distTotal)
    if distTotal > 0:
        incrementTotal = distTotal / DISCRETIZATION_STEP
        for i in range(0, N):
            dists[i] = dists[i] / incrementTotal

        numSegments = int(math.floor(incrementTotal))

        stateCurr = np.zeros(N, dtype=np.float32)
        for i in range(0, N):
            stateCurr[i] = start[i]
        for i in range(0, numSegments):

            if IsInCollision(stateCurr, idx):
                return 0

            for j in range(0, N):
                stateCurr[j] = stateCurr[j] + dists[j]

        if IsInCollision(end, idx):
            return 0

    return 1


steering_function = steerToPoly


# checks the feasibility of entire path including the path edges
def feasibility_check(path, idx):
    for i in range(0, len(path) - 1):
        ind = steering_function(path[i], path[i + 1], idx)
        if ind == 0:
            return 0
    return 1


# checks the feasibility of path nodes only
def collision_check(path, idx):
    for i in range(0, len(path)):
        if IsInCollision(path[i], idx):
            return 0
    return 1


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def get_input(i, dataset, targets, seq, bs):
    bi = np.zeros((bs, 2 * N), dtype=np.float32)
    bt = np.zeros((bs, N), dtype=np.float32)
    k = 0
    for b in range(i, i + bs):
        bi[k] = dataset[seq[i]].flatten()
        bt[k] = targets[seq[i]].flatten()
        k = k + 1
    return torch.from_numpy(bi), torch.from_numpy(bt)


def is_reaching_target(start1, start2):
    s1 = np.zeros(N, dtype=np.float32)
    for i in range(N):
        s1[i] = start1[i]

    s2 = np.zeros(N, dtype=np.float32)
    for i in range(N):
        s2[i] = start2[i]

    for i in range(0, N):
        if abs(s1[i] - s2[i]) > 1.0:
            return False
    return True


# lazy vertex contraction
def lvc(path, idx):
    for i in range(0, len(path) - 1):
        for j in range(len(path) - 1, i + 1, -1):
            ind = 0
            ind = steering_function(path[i], path[j], idx)
            if ind == 1:
                pc = []
                for k in range(0, i + 1):
                    pc.append(path[k])
                for k in range(j, len(path)):
                    pc.append(path[k])

                return lvc(pc, idx)

    return path


def re_iterate_path2(p, g, idx, obs):
    step = 0
    path = []
    path.append(p[0])
    for i in range(1, len(p) - 1):
        if not IsInCollision(p[i], idx):
            path.append(p[i])
    path.append(g)
    new_path = []
    for i in range(0, len(path) - 1):
        target_reached = False

        st = path[i]
        gl = path[i + 1]
        steer = steering_function(st, gl, idx)
        if steer == 1:
            new_path.append(st)
            new_path.append(gl)
        else:
            itr = 0
            target_reached = False
            while (not target_reached) and itr < 50:
                new_path.append(st)
                itr = itr + 1
                ip = torch.cat((obs, st, gl))
                ip = to_var(ip)
                st = mlp(ip)
                st = st.data.cpu()
                target_reached = is_reaching_target(st, gl)
            if target_reached == False:
                return 0

    # new_path.append(g)
    return new_path


def replan_path(p, g, idx):
    step = 0
    path = []
    path.append(p[0])
    for i in range(1, len(p) - 1):
        if not IsInCollision(p[i], idx):
            path.append(p[i])
    path.append(g)
    new_path = []
    for i in range(0, len(path) - 1):
        target_reached = False

        st = path[i]
        gl = path[i + 1]
        steer = steering_function(st, gl, idx)
        if steer == 1:
            new_path.append(st)
            new_path.append(gl)
        else:
            itr = 0
            pA = []
            pA.append(st)
            pB = []
            pB.append(gl)
            target_reached = 0
            tree = 0
            while target_reached == 0 and itr < 50:
                itr = itr + 1
                if tree == 0:
                    ip1 = torch.cat((st, gl))
                    ip1 = to_var(ip1)
                    st = mlp(ip1)
                    st = st.data.cpu()
                    pA.append(st)
                    tree = 1
                else:
                    ip2 = torch.cat((gl, st))
                    ip2 = to_var(ip2)
                    gl = mlp(ip2)
                    gl = gl.data.cpu()
                    pB.append(gl)
                    tree = 0
                target_reached = steering_function(st, gl, idx)
            if target_reached == 0:
                return 0
            else:
                for p1 in range(0, len(pA)):
                    new_path.append(pA[p1])
                for p2 in range(len(pB) - 1, -1, -1):
                    new_path.append(pB[p2])

    return new_path


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    tp = 0
    fp = 0
    tot = []
    times = []
    for i in range(len(paths)):
        et = []
        print(f"path: i={i}")
        p1_ind = 0
        p2_ind = 0
        p_ind = 0
        start = paths[i][0]
        # IsInCollision(start, 0)
        goal = paths[i][path_lengths[i] - 1]
        # start and goal for bidirectional generation
        ## starting point
        start1 = torch.from_numpy(start)
        goal2 = torch.from_numpy(start)
        ##goal point
        goal1 = torch.from_numpy(goal)
        start2 = torch.from_numpy(goal)
        ##generated paths
        path1 = []
        path1.append(start1)
        path2 = []
        path2.append(start2)
        path = []
        target_reached = 0
        step = 0
        path = []  # stores end2end path by concatenating path1 and path2
        tree = 0
        tic = time.perf_counter()
        # while target_reached == 0 and step < 1000:
        while target_reached == 0 and step < 500:
            # while target_reached == 0 and step < 80:
            step = step + 1
            if tree == 0:
                inp1 = torch.cat((start1, start2))
                inp1 = to_var(inp1)
                start1 = mlp(inp1)
                start1 = start1.data.cpu()
                path1.append(start1)
                tree = 1
            else:
                inp2 = torch.cat((start2, start1))
                inp2 = to_var(inp2)
                start2 = mlp(inp2)
                start2 = start2.data.cpu()
                path2.append(start2)
                tree = 0
            target_reached = steering_function(start1, start2, i)
        tp = tp + 1

        if target_reached == 1:
            for p1 in range(0, len(path1)):
                path.append(path1[p1])
            for p2 in range(len(path2) - 1, -1, -1):
                path.append(path2[p2])

            path = lvc(path, i)
            indicator = feasibility_check(path, i)
            if indicator == 1:
                toc = time.perf_counter()
                t = toc - tic
                print("TIME:", t)
                print("STEP:", step)
                et.append(t)
                times.append(t)
                fp = fp + 1
                print("path[0]:")
                plot_path(path)
                for p in range(0, len(path)):
                    print(path[p])
                # print("Actual path[0]:")
                # for p in range(0, path_lengths[i]):
                #    print(paths[i][p])
            else:
                sp = 0
                indicator = 0
                while indicator == 0 and sp < 10 and path != 0:
                    sp = sp + 1
                    g = np.zeros(2, dtype=np.float32)
                    g = torch.from_numpy(paths[i][path_lengths[i] - 1])
                    path = replan_path(path, g, i)  # replanning at coarse level
                    if path != 0:
                        path = lvc(path, i)
                        indicator = feasibility_check(path, i)

                    if indicator == 1:
                        toc = time.perf_counter()
                        t = toc - tic
                        et.append(t)
                        fp = fp + 1
                        if len(path) < 20:
                            print("new_path[0]:")
                            for p in range(0, len(path)):
                                print(path[p])
                            # print("Actual path:")
                            # for p in range(0, path_lengths[i]):
                            #    print(paths[i][p])
                        else:
                            print("path found, dont worry")

        tot.append(et)
    # pickle.dump(tot, open("time_s2D_unseen_mlp.p", "wb"))

    print("total paths")
    print(tp)
    print("feasible paths")
    print(fp)
    print("TIME")
    print(np.mean(times))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--no_env', type=int, default=50, help='directory for obstacle images')
    parser.add_argument('--no_motion_paths', type=int, default=2000, help='number of optimal paths in each environment')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--input_size', type=int, default=68, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int, default=2, help='dimension of the input vector')
    parser.add_argument('--hidden_size', type=int, default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
